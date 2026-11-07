"""Search environment: the tag protocol, run at the token level.

Each turn the model writes free text that ends in exactly one action:
    <search>query</search>   the environment appends a <result> block with the top 3 BM25 passages,
                             then a one-line reminder to act rather than answer (without it the
                             1.5B model answers the question after the first result: format
                             validity 32% -> 94% on 200 train questions)
    <done/>                  the episode ends
The episode also ends after the 3rd search (its results still count) or on a malformed turn.
The final set is the union of all returned passages, deduplicated in order (at most 9).

Malformed turn (episode ends, retrieved passages still count, format reward not paid):
  - no action before the turn's token cap or end-of-turn token. An unclosed <search> at the cap
    is malformed too: one rule for the prompted and the trained model, no silent repair.
  - a model-written <result> (tag mimicry; generation stops on it)
  - an empty query, or a query longer than MAX_QUERY_WORDS (query stuffing)

Option separate_results (default off; core-v1 is inline): close the assistant turn after
</search>, put the result block (and reminder) in its own user message via the chat template,
and open a new assistant turn. Option reminder=False drops the reminder line. Option system_note (default "", core-v1) appends
one paragraph to the system prompt (Exp 05: what the model is told about its grader). Option
harness (default CORE, Exp 06) swaps the system prompt, the two action tokens and the reminder for
a harness the model was not trained in; W rewords the prompt, T also renames the tokens
(<query>...</query> / <stop/>). The <result> block tag is the same in every harness.

Chat templates that take enable_thinking (Qwen3, Exp 07) get enable_thinking=False, so the turn
budget goes to the action; templates without it (Qwen2.5) get no extra kwarg and their prompts stay
byte-identical to core-v1. Each turn records n_tokens, the number of generated tokens.

The environment keeps the full token sequence and a mask that is 1 on model-generated tokens and
0 on prompt and environment tokens. The trainer uses exactly these ids and this mask.

Reward: R = recall + 0.2 * format_valid - 0.05 * n_searches, where format_valid also needs at
least one search (no farming the format reward by stopping at once). See plan/05-reward.md.
"""

import re
import threading

from nanosearch.indexes import load_corpus

MAX_SEARCHES = 3
PASSAGES_PER_SEARCH = 3
TURN_TOKENS = 256
PASSAGE_WORDS = 100
MAX_QUERY_WORDS = 32
STOP = ["</search>", "<done/>", "<result>"]
REMINDER = "Do not answer the question. End this turn with <search>...</search> or <done/>.\n"
W_FORMAT, W_STEP = 0.2, 0.05

SYSTEM_PROMPT = """You are a search agent. Your job is to find the Wikipedia passages needed to answer a multi-hop question. You do not answer the question. You use a keyword search engine (BM25) over Wikipedia passages.

Each turn, think in one or two sentences, then end the turn with exactly one action:
<search>keywords</search>   returns the top 3 passages
<done/>                     stop, when you have found every passage the question needs

Never write the answer to the question. Every turn must end with <search>...</search> or <done/>.
You can search at most 3 times. A good query names the specific entity you need next, using what you learned from earlier results. Do not repeat a query.

Example:
Question: When was the university attended by the author of Things Fall Apart founded?
First I need the author of Things Fall Apart. <search>Things Fall Apart novel author</search>
<result>
[1] Things Fall Apart | Things Fall Apart is a novel by Nigerian author Chinua Achebe, published in 1958...
[2] Chinua Achebe | Chinua Achebe was a Nigerian novelist. He studied at University College, Ibadan...
[3] No Longer at Ease | No Longer at Ease is a 1960 novel by Chinua Achebe...
</result>
The author is Chinua Achebe, who studied at University College, Ibadan. Now I need when that university was founded. <search>University College Ibadan founded</search>
<result>
[1] University of Ibadan | The University of Ibadan is the oldest university in Nigeria, founded in 1948 as University College, Ibadan...
[2] Ibadan | Ibadan is the capital of Oyo State...
[3] Obafemi Awolowo University | Obafemi Awolowo University was founded in 1962...
</result>
I have the novel's author and the university's founding year. <done/>"""

ACTION_RE = re.compile(r"<search>(.*?)</search>|<done/>", re.S)


class Harness:
    """System prompt, action tokens and reminder of one interaction protocol (Exp 06)."""

    def __init__(self, name, system, search_open, search_close, done, reminder):
        self.name, self.system = name, system
        self.search_open, self.search_close, self.done = search_open, search_close, done
        self.reminder = reminder
        self.stop = [search_close, done, "<result>"]
        self.action_re = re.compile(
            f"{re.escape(search_open)}(.*?){re.escape(search_close)}|{re.escape(done)}", re.S)

    def __repr__(self):
        return f"Harness({self.name})"


CORE = Harness("core", SYSTEM_PROMPT, "<search>", "</search>", "<done/>", REMINDER)
assert CORE.stop == STOP and CORE.action_re.pattern == ACTION_RE.pattern

# Exp 06 harness W: paraphrased prompt, same tokens, different worked example. Verbatim from
# experiments/06-harness-transfer/README.md.
W_SYSTEM_PROMPT = """You search Wikipedia for a reader who has a question that needs facts from more than one passage.
Your task is only to collect those passages, not to state the answer. The search engine is BM25,
so it matches keywords.

In each turn write one or two sentences about what you still need, then finish the turn with a
single action:
<search>keywords</search>   shows you the 3 best-matching passages
<done/>                     ends the search, once every passage the reader needs is in hand

Do not state the answer to the question. Each turn has to finish with <search>...</search> or <done/>.
At most 3 searches are allowed. Good keywords name the exact thing you still need, based on what
the earlier results told you. Never send the same query twice.

Example:
Question: How many inhabitants does the town have in which the composer of the opera Nabucco was born?
The opera is Nabucco, so first I need its composer. <search>Nabucco opera composer</search>
<result>
[1] Nabucco | Nabucco is an opera in four acts by Giuseppe Verdi, first performed in Milan in 1842...
[2] Giuseppe Verdi | Giuseppe Verdi was an Italian composer, born in Le Roncole, a village in the municipality of Busseto...
[3] Va, pensiero | Va, pensiero is a chorus from the opera Nabucco...
</result>
Verdi was born in Le Roncole, which belongs to Busseto. Now I need the population of Busseto. <search>Busseto population</search>
<result>
[1] Busseto | Busseto is a comune in the province of Parma, Emilia-Romagna, with about 7,000 inhabitants...
[2] Province of Parma | The Province of Parma is a province in Emilia-Romagna...
[3] Le Roncole | Le Roncole is a village in the comune of Busseto, known as Verdi's birthplace...
</result>
I have the composer's birthplace and the town's population. <done/>"""

W = Harness("W", W_SYSTEM_PROMPT, "<search>", "</search>", "<done/>", REMINDER)
# Exp 06 harness T: the W prompt with renamed action tokens; used with separate_results.
T = Harness("T", W_SYSTEM_PROMPT.replace("<search>", "<query>").replace("</search>", "</query>")
            .replace("<done/>", "<stop/>"), "<query>", "</query>", "<stop/>",
            "Do not answer the question. End this turn with <query>...</query> or <stop/>.\n")
HARNESSES = {"core": CORE, "W": W, "T": T}


def reward(recall, format_valid, n_searches, w_step=W_STEP):
    return recall + W_FORMAT * format_valid * (n_searches > 0) - w_step * n_searches


def judge_score(turns, gold_titles):
    """Exp 04 rule-based "summary judge": share of the gold titles that appear (case-insensitive
    substring) in the episode's final turn. It reads only that text, never the retrieved passages.
    If the episode hit the search cap without <done/>, the final turn is the last search turn."""
    text = turns[-1]["text"].lower() if turns else ""
    return sum(t.lower() in text for t in gold_titles) / len(gold_titles)


def unsupported_titles(turns, gold_titles, retrieved_titles):
    """(gold titles named in the final turn, those of them not among the retrieved titles)."""
    text = turns[-1]["text"].lower() if turns else ""
    named = [t for t in gold_titles if t.lower() in text]
    got = {t.lower() for t in retrieved_titles}
    return len(named), sum(t.lower() not in got for t in named)


def parse_turn(text, harness=CORE):
    """-> ("search", query) | ("done", None) | ("malformed", reason)"""
    if "<result>" in text:
        return "malformed", "tag_mimicry"
    m = harness.action_re.search(text)
    if m is None:
        return "malformed", "no_action"
    if m.group(0) == harness.done:
        return "done", None
    query = m.group(1).strip()
    if not query:
        return "malformed", "empty_query"
    if len(query.split()) > MAX_QUERY_WORDS:
        return "malformed", "query_too_long"
    return "search", query


class SearchEnv:
    """Shared, thread-safe: holds the tokenizer, the retriever and the passage texts."""

    def __init__(self, tokenizer, retriever, corpus_path=None, separate_results=False,
                 reminder=True, system_note="", harness=CORE):
        self.tok = tokenizer
        self.retriever = retriever
        self.separate_results, self.reminder = separate_results, reminder
        self.harness = harness
        self.system = harness.system + (f"\n\n{system_note}" if system_note else "")
        template = getattr(tokenizer, "chat_template", None) or ""
        self.template_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
        if separate_results:
            # The text the chat template puts between an assistant message and the next user
            # message, and between that user message and the next assistant turn.
            a, u = "\x00ASSISTANT\x00", "\x00USER\x00"
            s = tokenizer.apply_chat_template(
                [{"role": "user", "content": "q"}, {"role": "assistant", "content": a},
                 {"role": "user", "content": u}], add_generation_prompt=True, tokenize=False,
                **self.template_kwargs)
            self.close_turn, self.open_turn = s.split(a)[1].split(u)
        docs = load_corpus(corpus_path) if corpus_path else load_corpus()
        self.docs = {d["doc_id"]: d for d in docs}
        self.lock = threading.Lock()

    def prompt_ids(self, question):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": f"Question: {question}"}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            **self.template_kwargs)

    def result_block(self, doc_ids):
        lines = []
        for i, d in enumerate(doc_ids, 1):
            words = self.docs[d]["text"].split()
            text = " ".join(words[:PASSAGE_WORDS]) + ("..." if len(words) > PASSAGE_WORDS else "")
            lines.append(f"[{i}] {self.docs[d]['title']} | {text}")
        block = ("<result>\n" + "\n".join(lines) + "\n</result>\n"
                 + (self.harness.reminder if self.reminder else ""))
        if self.separate_results:
            return self.close_turn + block.rstrip("\n") + self.open_turn
        return "\n" + block

    def run(self, question, generate):
        """Play one episode. generate(prompt_ids, max_tokens) -> generated token ids."""
        ids = list(self.prompt_ids(question))
        mask = [0] * len(ids)
        queries, retrieved, turns = [], [], []
        end, n_generated = "max_searches", 0
        while True:
            gen = generate(ids, TURN_TOKENS)
            ids += gen
            mask += [1] * len(gen)
            n_generated += len(gen)
            text = self.tok.decode(gen, skip_special_tokens=True)
            kind, arg = parse_turn(text, self.harness)
            turns.append({"text": text, "action": kind, "arg": arg, "n_tokens": len(gen)})
            if kind != "search":
                end = "done" if kind == "done" else f"malformed:{arg}"
                break
            queries.append(arg)
            with self.lock:
                hits = self.retriever.search(arg, PASSAGES_PER_SEARCH)
            retrieved += [d for d in hits if d not in retrieved]
            if len(queries) == MAX_SEARCHES:
                break
            env_ids = self.tok.encode(self.result_block(hits), add_special_tokens=False)
            ids += env_ids
            mask += [0] * len(env_ids)
        return {"ids": ids, "mask": mask, "queries": queries, "retrieved": retrieved,
                "turns": turns, "end": end, "format_valid": int(not end.startswith("malformed")),
                "n_searches": len(queries), "n_generated": n_generated}

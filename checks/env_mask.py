"""Check the env's token sequence and mask in all four result layouts, with a scripted policy.

The mask must be 1 exactly on the scripted model tokens and 0 on everything the env adds, and in
separate_results mode the sequence must equal the chat template rendering of the same messages.

  python -m checks.env_mask
"""

from transformers import AutoTokenizer

from nanosearch.agent import MODEL
from nanosearch.env import SYSTEM_PROMPT, SearchEnv


class FakeRetriever:
    def search(self, query, k):
        return ["d00000", "d00001", "d00002"] if "first" in query else ["d00003", "d00004", "d00005"]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    turns = ["I need X. <search>first query</search>", "Now Y. <search>second query</search>",
             "I have both. <done/>"]
    for sep in (False, True):
        for rem in (True, False):
            env = SearchEnv(tok, FakeRetriever(), separate_results=sep, reminder=rem)
            script = iter(turns)
            gen_ids = []

            def generate(ids, max_tokens):
                g = tok.encode(next(script), add_special_tokens=False)
                gen_ids.append(g)
                return g

            ep = env.run("Who?", generate)
            ids, mask = ep["ids"], ep["mask"]
            assert len(ids) == len(mask) and ep["end"] == "done" and ep["n_searches"] == 2
            assert sum(mask) == sum(map(len, gen_ids)) == ep["n_generated"]
            # mask-1 positions are exactly the scripted tokens, in order
            assert [t for t, m in zip(ids, mask) if m] == [t for g in gen_ids for t in g]
            # every mask-1 run is exactly one generated turn
            runs, cur = [], []
            for t, m in zip(ids, mask):
                if m:
                    cur.append(t)
                elif cur:
                    runs.append(cur); cur = []
            runs.append(cur)
            assert runs == gen_ids, "env token leaked into a model run"
            text = tok.decode(ids)
            assert text.count("<result>") - SYSTEM_PROMPT.count("<result>") == 2
            assert text.count("Do not answer") - SYSTEM_PROMPT.count("Do not answer") == 2 * rem
            if sep:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Question: Who?"}]
                for t, hits in zip(turns, [["d00000", "d00001", "d00002"], ["d00003", "d00004", "d00005"]]):
                    msgs += [{"role": "assistant", "content": t},
                             {"role": "user", "content": env.result_block(hits)
                              [len(env.close_turn):-len(env.open_turn)]}]
                ref = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                assert text == ref + turns[2], "separate mode differs from the chat template"
                assert text.count("<|im_start|>assistant") == 3
            else:
                assert text.count("<|im_start|>assistant") == 1
            print(f"separate={sep} reminder={rem}: OK, {len(ids)} tokens, {sum(mask)} masked in")
    print(repr(tok.decode(ids)[-700:]))


if __name__ == "__main__":
    main()

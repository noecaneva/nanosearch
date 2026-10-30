# 07. Technical Report Outline

The report is not documentation of the code. It is half the deliverable, and it is the half that
shows the work can be explained clearly. A big part of making successful models is telling people
about them.

Target five to eight pages. Write it as though the reader trains models for a living and has
fifteen minutes.

## Structure

**1. What and why, half a page.** The question the project answers, the headline number, and the
scale caveat stated immediately rather than buried. A reader who stops here should still have the
result.

**2. Setup, one page.** Task, corpus construction with real counts, action space, and the reward.
Enough for a reader to rebuild it. The corpus deduplication rule belongs here, not in an appendix.

**3. Method, one to two pages.** GRPO as implemented, including the advantage computation and the
clipped objective. Then the two implementation details that mattered: masking environment tokens,
and handling zero-variance groups. Say what went wrong before you got them right. Both are
non-obvious, both are where an expert reader's attention will be, and having debugged them is the
thing being demonstrated.

**4. Results, one to two pages.** The four-way table with recall, nDCG, mean searches, latency and
token cost. Then the ablations. Include the latency and cost columns even where they are
unflattering.

**5. What the model did to cheat, one page.** The hacking register with real examples of episodes.
Verbatim rollouts, including the embarrassing ones. This is the section a research engineer will
read first and the one most candidates do not write at all.

**6. Limitations, half a page.** Model two orders of magnitude below production scale. Corpus three
orders below. Tag protocol rather than real tool calling. No vision. Single training run per
configuration with no seed variance, if that is the case. State these before a reader finds them.

**7. What I would do next, half a page.** Concrete and specific: which ablation is missing, what
would change at 7B, what the selection action space would need, where the compute went.

## Rules for the writing

Numbers in the abstract. State the scale caveat in the first paragraph, never as a footnote. Every
claim either measured or marked as speculation. Show at least three real rollouts in full, chosen to
include one good, one degenerate, and one that is just strange.

No em dashes. No sentence that starts a paragraph and then stops after five words.

If the result is negative, the title says so. "Why GRPO did not beat prompting at 1.5B" is a
stronger artifact than a vague title hiding the same content.

## Publishing

Repo public on GitHub with a README a stranger can follow, the report as a PDF in the repo, and a
short thread or post pointing at it. The point of the project is that someone who trains models can read it in
fifteen minutes and form an accurate view of how you work.

# Paper Writing Assistant (SCI/IEEE Conference)

## Writing Order Principle (前戏后做 / 主体先行)

**Write in this order — not submission order:**

1. **Methods** — describe your approach precisely
2. **Experiments** — present your setup and results
3. **Results/Analysis** — interpret findings, ablations, discussion
4. **Conclusion** — summarize contributions and limitations
5. **Introduction** — write LAST among body sections (depends on knowing what you proved)
6. **Abstract** — write VERY LAST (distills the complete paper)
7. **Title** — finalize after abstract

Rationale: Abstract and Introduction frame what the paper delivers. You can only write them accurately after you know what the paper actually shows.

---

## Section Templates

### Abstract (5 components)

```
[1] BACKGROUND + GOAL: What domain/problem motivates this work? What is the high-level objective?
[2] PROBLEM: What specific gap or challenge does this paper address? (1-2 sentences)
[3] CONTENT + METHODS: What did you do? Describe the proposed method/approach briefly.
[4] RESULTS: Key quantitative outcomes. State numbers. Compare to baselines.
[5] APPLICATION/SIGNIFICANCE: What does this enable? Why does it matter?
```

**CS/engineering papers**: Omit limitations from the abstract (save for Experiments/Conclusion).
**Length**: 150–250 words. One dense paragraph. No citations.

**Template fill-in:**
```
[Background] X is critical for Y, but [problem]. Existing methods [limitation].
We propose [method name], which [core mechanism].
[Method] achieves [key result: XX% on Dataset vs YY% baseline].
[Significance] This enables/demonstrates [broader impact].
```

---

### Introduction (4 components)

```
[1] IMPORTANCE + BACKGROUND: Why does this problem matter? Establish domain context.
    → Open with compelling fact, statistic, or application importance.
    → 2-4 sentences establishing why readers should care.

[2] PRIOR WORK + CURRENT STATE: What have others done? What is the current best approach?
    → Survey key methods (with citations). 
    → Acknowledge their contributions, then pivot to what they miss.
    → Do NOT trash prior work — acknowledge it, then show the gap.

[3] RESEARCH GAP / MOTIVATION: What specific problem remains unsolved?
    → "However, [limitation of prior work]."
    → "This leaves open the question of [your research question]."
    → This is the hinge sentence. Make it crisp and specific.

[4] CONTRIBUTIONS: What does this paper contribute?
    → Use a bulleted list of 3-5 contributions.
    → Each should be concrete and verifiable.
    → Close with paper organization: "The rest of this paper is organized as..."
```

**Sentence-level pattern** (from Normal-GS analysis):
- S1-2: Domain importance (hook)
- S3-5: Prior work landscape (citations)
- S6-7: Remaining challenge ("However, ...")
- S8-10: Your approach overview
- S11+: Contribution list

**Do NOT**: Over-cite in opening sentences. Save dense citations for the prior work paragraph.

---

### Related Work

```
[1] ORGANIZE BY THEME: Group papers into 2-4 thematic subsections, not chronologically.
[2] COMPARE, DON'T CATALOG: For each group, state what the methods share, then what distinguishes yours.
[3] POSITION: The last sentence of Related Work should make clear why prior work is insufficient and yours is needed.
```

**Typical subsections** for a medical image segmentation paper:
- Transformer-based segmentation
- Attention mechanisms (global vs local)
- Efficient attention approximations
- [Task-specific: e.g., polyp/skin lesion segmentation]

---

### Methods (4 components)

```
[1] GENERAL OVERVIEW: What is the overall architecture / approach? A bird's-eye view.
    → Include a system diagram (Figure 1 is almost always architecture).
    → One paragraph: input → module → output.

[2] PRECISE DETAILS: Mathematical formulation. Component-by-component description.
    → Equations for key operations.
    → For each module: what it takes as input, what it outputs, what it computes.
    → Use consistent notation: define variables once, reuse everywhere.

[3] LINK TO PRIOR RESEARCH: How does your design relate to or differ from prior methods?
    → "Unlike [X], which [limitation], our [component] [advantage]."
    → This justifies your design choices.

[4] PROBLEMS ADDRESSED: What specific problem does each design choice solve?
    → For each key design decision: state the problem it solves and why your choice is correct.
    → This is where you pre-empt reviewer questions.
```

**Equation conventions:**
- Define all symbols at first use
- Reference every equation in text ("As shown in Eq. (3)...")
- Use \mathbf for vectors/matrices, plain italic for scalars

---

### Experiments (4 components)

```
[1] REVISIT OBJECTIVES: Restate what you set out to show.
    → "We evaluate [method] on [datasets] to answer the following questions: (1)... (2)... (3)..."
    → Maps experiments to claims in introduction.

[2] VIEW RESULTS: Present tables/figures. State findings plainly.
    → Lead with the headline number: "ApproxDA-TransUNet achieves 78.64% DSC on Synapse, ..."
    → Compare to baselines row-by-row for key metrics.
    → Call out surprising or particularly strong results.

[3] PROBLEMS WITH RESULTS / ANALYSIS: What do the results reveal beyond the headline?
    → Ablation study: which component contributes what?
    → Error analysis: where does the method fail?
    → "Interestingly, [observation]" → explain why.

[4] IMPLICATIONS: What do the results tell us about the method / problem?
    → Connect back to research gap from Introduction.
    → "These results suggest that [insight about the problem]."
    → Limitations: where does the method not work? (CS papers put this here, not in Abstract/Conclusion)
```

**Table formatting conventions:**
- Bold the **best** result per column
- Underline the second-best
- Use ↑/↓ symbols to indicate metric direction in column headers
- Report mean ± std if results vary across runs

---

### Conclusion (4 components)

```
[1] REVIEW: One-paragraph summary of what was done and what was found.
    → Do NOT just copy the abstract. Synthesize at a higher level.

[2] POSITIONING: Where does this work sit in the broader field?
    → "This work establishes [X] as a viable approach for [Y]."
    → Link to the research gap you stated in the Introduction.

[3] CONTRIBUTIONS (restated briefly): What are the 2-3 take-home contributions?
    → Shorter than the Introduction list. Just the headline findings.

[4] LIMITATIONS + FUTURE WORK: What doesn't this paper solve?
    → Be honest. Reviewers respect candor.
    → Frame as "Future work will [X]" — turns limitations into opportunities.
```

**CS/engineering papers**: Limitations can also appear at the end of Experiments. Conclusion then omits them or just references the Experiments limitation paragraph.

---

## Figure Planning (5-figure structure for conference paper)

```
Figure 1: Architecture overview (Methods) — always the first figure
Figure 2: Key analytical result (e.g., gate collapse visualization, attention maps)
Figure 3: Main quantitative comparison — bar chart across datasets/methods
Figure 4: Ablation results — table or bar chart showing component contribution
Figure 5: Qualitative case visualization — side-by-side prediction vs. ground truth
```

---

## Common Mistakes to Avoid

| Mistake | Fix |
|---------|-----|
| Writing Abstract first | Write it last — you don't know what you proved yet |
| Vague contribution claims ("We propose a novel...") | Be specific ("We prove that gate collapse is inevitable when PAM≈CAM due to gradient symmetry") |
| Citing in the opening sentence | Hook first, citations second paragraph |
| Copying abstract into conclusion | Synthesize at a higher level in conclusion |
| Passive voice overuse | Mix active/passive — active for your contributions, passive for prior work |
| Burying the key result | State the headline number in sentence 2 of the abstract and introduction |
| Introducing symbols without definition | Define every variable at first use |

---

## ApproxDA-TransUNet Paper Outline (Reference)

**Paper title**: "Understanding Context-Dependent Attention Approximation for Medical Image Segmentation"
**Method name**: ApproxDA-TransUNet (Approximate Dual Attention TransUNet)
**Target**: BIBM 2026

## Usage

When asked to draft a paper section, use this structure:
1. Identify which section is requested
2. Ask for the content inputs (what results/methods to describe) if not provided
3. Apply the 4-component template for that section
4. Check: does each component appear? Is the writing order correct?
5. For the full paper: confirm the user is writing Methods first, not Abstract first

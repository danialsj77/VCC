# Draft: rules questions for the organizers (task #7)

For **Daniyal** to send himself. Nothing has been sent.

- **To:** help@virtualcellchallenge.org
- **From:** the address Daniyal registered with (the organizers match questions to registered accounts).
- **Before sending:** replace `<TEAM NAME>` (3 places) with our team's name as the challenge web app shows it.
  Nothing else needs filling in.
- **Send it only by e-mail.** Don't post it on Discord: it describes our method, and other teams read Discord.
- **After sending:** note the date in CONTRIBUTING.md §8 (task #7). When the answers arrive, save them in
  `notes/organizer_answers.md`, then update METHOD.md §5 and the generator choice in RUNBOOK.md §7.

Why each question matters:

| # | Decides |
|---|---|
| 1 | Whether model v1 (same-target K562 transfer with fitted gains) may be our final entry, or only the learned transfer model (v2) may |
| 2 | Generator for the final entry: control-cell templates (`--allow-control-templates`) or the NB generator (current default) |
| 3, 4 | Which public models and datasets a prize-eligible entry may use |
| 5 | Whether v2 may weight source lines by similarity to the new line's control profile |
| 6, 7 | How we plan the final phase (no feedback is expected) |
| 8 | What the finalist method description has to contain |

The text below is ready to paste. Its paragraphs are deliberately not wrapped, so the mail client can wrap them.

---

**Subject:** VCC 2026 rules questions (team <TEAM NAME>): cross-line transfer, control-cell templates, licences

```text
Hello VCC team,

I lead team <TEAM NAME> and am writing from the e-mail address I registered with. Before the final phase opens on October 22, could you confirm the following points under the 16 September rules revision?

1. Cross-line transfer of public responses. Our current model predicts a target's response in a challenge context from the same target's measured response in a public Perturb-seq screen of another cell line (Replogle et al. 2022, K562 genome-wide). The published log fold changes are denoised with a z-score threshold, rescaled by two gains fitted by least squares on a held-out line (Arc's VCC 2025 H1 data), and combined with the screen's average response over the panel and a prior for CRISPRi silencing of neighbouring genes that depends only on promoter distance. Is this a machine-learning prediction trained on public data, or does it count as submitting measured results, or using them to revise model outputs? Does the answer change if the mapping is a regression with many learned parameters (for example ridge regression or gradient boosting over the source responses and features of the context's control cells) instead of two global gains? And if a challenge cell line turns out to be, or closely resemble, a line with public screen data, may that data still be a model input at inference time?

2. Control cells in the cell generator. To produce 400 count cells per target, one of our generators resamples the provided control cells of the same context and edits them: binomial thinning for genes predicted to go down, added Poisson counts for genes predicted to go up. Genes predicted to be unchanged keep the control cell's counts. The data page says the controls are "model inputs only — do not copy them into your prediction". Is this generator allowed? Or must every count be sampled from a model fitted to the controls? Our alternative samples every count from a negative binomial model fitted per gene to the controls, with each generated cell taking the library size of a randomly drawn control cell. Is reusing the control cells' library sizes in this way acceptable?

3. Is Arc's PIE covered by the same policy as State and Stack for prize-eligible entries, given its separate non-commercial licence?

4. May non-commercial datasets (for example X-Atlas/Orion, CC BY-NC-SA) and Arc's VCC 2025 H1 data be used to train a prize-eligible entry?

5. May a model infer a context's lineage, or its similarity to public cell lines, from its control-cell profile, and use that to weight training sources?

6. In the final phase, will teams see whether each entry was scored successfully, and its scores?

7. Will rule_version 4 (the MSE change discussed in cell-eval2 issues #294 and #348) score the final phase, and will it be in place before the D/E/F data opens on October 22?

8. Our model uses a few non-learned components: a fixed knockdown size for the target's own gene, a promoter-distance prior for neighbouring genes, an expression floor below which no change is predicted, a minimum effect size for confidently changed genes, and thresholds and scale factors chosen by cross-validation on held-out data. Is listing them in the finalist method description sufficient?

Thank you for your help.

Best regards,
Daniyal Sajedi Hosseini
Team lead, <TEAM NAME>
```

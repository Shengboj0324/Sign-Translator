# Gloss review packet

For qualified ASL reviewers. Each row is one 3D-LEX sign label with its ASL Signbank
candidates (`ID|GLOSS|URL`), hardest cases first: 195 with several
variant candidates, 59 with keyword matches only, 744 with
one stem candidate, 3 unmatched.

Fill `decision` with one of `confirm`, `choose`, `none_match`, `unsure`; for
`confirm`/`choose` put an offered candidate's ID in `chosen_cve_id`. Put a SHA-256 key
in `reviewer_sha256`, never a name. Leave other columns unchanged. Returned sheets are
read by `data_engineering/gloss_review.parse_review_sheet`, which refuses anything
else, and two reviewers' agreement is measured by `review_agreement` (Cohen's kappa).

Source manifest SHA-256 `c761728913baf7be422a665c377952852ae09504b444d92e746081ab819561e7`;
packet SHA-256 `6293670343b0ce65006b8ee25e12d5cbae5a3344cb30583bc7a91d31a0312068`. Derived from ASL Signbank
(CC BY-NC-SA 4.0): non-commercial, share alike; cite Hochgesang, Crasborn and
Lillo-Martin, ASL Signbank, doi:10.6084/m9.figshare.9741788.v2.

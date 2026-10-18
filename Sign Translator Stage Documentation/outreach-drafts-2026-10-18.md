# Outreach drafts — 2026-10-18 (not sent)

These drafts unblock the items that need people outside the project. **None has been
sent.** Fill the bracketed fields, review the wording, and send them yourself. Addresses
are the ones recorded in the acquisition guide; where none is recorded, the route is
named instead of guessed.

---

## 1. RIT/CUNY continuous motion sample (closes B01 if the sample qualifies)

**To:** matt.huenerfauth@rit.edu
**Subject:** Research access request: one continuous ASL motion-capture sample

Dear Dr. Huenerfauth,

I am [name, role, affiliation] building a research ASL generation system whose data
layer requires continuous signing with body, both hands, face and gaze captured on
one clock. Your lab's corpus release (latlab.ist.rit.edu/downloads.html#corpus) is the
closest match we have identified.

Could we request access to one complete multi-sentence recording first: the motion
files in their native format, any calibration and clock documentation, the linguistic
annotation for that recording, and the applicable terms? We would validate our
adapter against it before asking for more. We use the data for non-commercial
research only, keep signer identity as salted hashes, and never redistribute media.

If a data use agreement is required, I would be glad to complete it.

With thanks,
[name]

---

## 2. 3D-LEX authors: synchronisation, missing files and two export defects

**Route:** the corresponding author of the 3D-LEX paper
(aclanthology.org/2024.signlang-1.33), or the issue tracker at
github.com/OlineRanum/SAPA. No address is recorded in the guide.
**Subject:** 3D-LEX v1.0 ASL: face/skeleton clock mapping and export observations

Dear 3D-LEX authors,

Thank you for releasing 3D-LEX v1.0 ASL under CC BY 4.0. While converting it we
measured the following, and would value your view:

1. **Face-to-skeleton synchronisation.** The Live Link Face CSVs carry device
   timecode, while the FBX take start times differ from them by an hour plus a few
   seconds that vary between takes. With head rotation recorded as zero (a head-worn
   device), we cannot align face and body ourselves. Is there a timecode mapping or a
   shared event per take?
2. **Missing left-glove exports.** 195 signs have only a right-hand StretchSense
   export. Are these one-handed signs recorded with the dominant glove only?
3. **Rest-pose final keys.** In 181 files from one export batch, the last key returns
   every joint to the bind pose.
4. **Marker trials.** For one performer, marker trials are often shorter than the
   animation (median 1.82 s against 5.33 s) and no single clock aligns them.

We handle each conservatively and can share the per-file lists if useful.

With thanks,
[name, affiliation]

---

## 3. Qualified gloss review (closes B74)

**To:** motionlightlab@gallaudet.edu (or a CARD/ASLLRP contact)
**Subject:** Paid review of 1,001 isolated-sign gloss matches

Dear Motion Light Lab team,

We have 1,001 isolated ASL signs from the public 3D-LEX lexicon, each matched by its
English label to ASL Signbank ID-gloss candidates. 195 have several variant candidates
and need a qualified signer to choose; 744 have a single candidate to confirm. The
review sheet (`evidence/gloss-review-packet-2026-10-18/review-packet.csv`) links every
candidate to its Signbank page.

Would a Deaf or qualified ASL reviewer, ideally two independently, be available for
paid review? Reviewers would be identified to us only by a key they choose, and
answers would be recorded with inter-reviewer agreement.

With thanks,
[name, affiliation]

---

## 4. Cokely HD video (B67) — your own download, no email needed

Open the publisher page in a normal browser, download
`2_I_Have_a_Dream_720_CokelyAFSParallelCorpus_v1_0.mp4`, and tell me its location.
I will check it against the expected 650,953,008 bytes and SHA-256
`8eb18b6b2f01a5a179100b0acd84a639b5f2a0ee95fb0195d477b946217c6bb3` before it is used.

# Legal Compliance & Data Ethics Guide

> **This document is mandatory reading for every contributor.** Submitting data
> or code that violates the rules below may expose you — and the project — to
> civil and criminal liability under U.S., E.U., and other national laws.

---

## 1. Scope

The Global UAP Intelligence Hub aggregates **publicly available, lawfully
obtained, unclassified** observational data about Unidentified Anomalous
Phenomena. The project explicitly does **not** accept:

- Classified material at any level (Confidential, Secret, Top Secret, SCI,
  SAP, or foreign equivalents).
- Information bearing **classified watermarks**, banner lines, portion marks,
  or `(U//FOUO)`, `(C)`, `(S//NOFORN)`, `(TS//SCI)` markings — even partially
  redacted.
- Material covered by the **International Traffic in Arms Regulations
  (ITAR — 22 CFR §§ 120–130)** or the **Export Administration Regulations
  (EAR — 15 CFR §§ 730–774)**.
- Defense data unlawfully obtained (e.g. leaked, hacked, exfiltrated, or
  obtained in violation of a non-disclosure agreement).
- Material whose release would violate the U.S. **Espionage Act
  (18 U.S.C. § 793 et seq.)** or equivalent statutes in any other jurisdiction.

If you are a current or former member of a military, intelligence, or
classified-program workforce, you are responsible for ensuring that any
material you contribute has been **formally declassified** and cleared for
public release by your originating agency's pre-publication review process.

---

## 2. Identifying ITAR / EAR-Controlled Material

ITAR controls **defense articles and services** enumerated on the U.S.
Munitions List (USML, 22 CFR § 121). EAR controls **dual-use items** on the
Commerce Control List (CCL, 15 CFR Part 774). The following are common red
flags. **If any of these appear in your submission, do not submit.**

### 2.1 Hardware / sensor data

- Performance metrics for **military-grade radar, IR, EO, or SIGINT sensors**
  (e.g. AN/APG-, AN/AAQ-, AN/ALR- series, AESA, LPI, ESM, SAR).
- Detailed waveform, frequency-hopping, or pulse-compression parameters.
- Inertial measurement unit (IMU) or guidance system parameters that exceed
  CCL/USML thresholds (e.g. gyro bias < 0.05 °/hr).
- Imagery with embedded classified metadata (NITF, KLV, MISB ST 0601 with
  classified security tags).

### 2.2 Aircraft / platform data

- Stealth signature measurements (RCS, IR), low-observable coatings, or
  shaping data.
- Flight envelopes, performance, or sensor fit of restricted platforms
  (F-22, F-35, B-2, B-21, RQ-170, RQ-180, etc.).
- Tactics, techniques, and procedures (TTPs) of military operators.

### 2.3 Cryptographic / RF / Software

- Type-1 cryptographic implementations or keys.
- Software with controlled cryptography exceeding EAR § 740.17 thresholds.
- Communications intelligence (COMINT) or measurement intelligence (MASINT)
  collection methods.

### 2.4 Documents

- Anything with a classification banner line, portion marks, "Controlled
  Unclassified Information (CUI)" markings, "Distribution Statement
  B/C/D/E/F", "NOFORN", "FRD", "RD", or "FGI".
- DD Form 254, DD Form 1540, DD Form 2024 attachments.
- Anything stamped "Export Controlled" or "ITAR Controlled".

> **When in doubt, leave it out.** If you cannot affirmatively confirm an item
> is unclassified and uncontrolled, do not submit it.

---

## 3. Submission Triage Procedure (for maintainers)

Maintainers reviewing a `data-report` issue or pull request **must**:

1. **Visually inspect** every attached file for classification banners,
   portion marks, or CUI labels before any download is propagated to other
   maintainers.
2. **Strip embedded metadata** (`exiftool -all=`) only *after* confirming the
   metadata itself does not contain classified pointers.
3. **Reject and close** any submission containing red-flag material. Use the
   template response in §6.
4. **Report** to the project's legal point of contact (see `CODEOWNERS`)
   within 24 hours if a submission appears to contain genuinely classified
   data. Do **not** redistribute the material while the report is pending.
5. **Never** screenshot, quote, or re-host classified material — including in
   issue comments — even to "show" what was rejected.

---

## 4. Contributor Code of Conduct (Data Ethics)

By submitting data or code, you agree that:

1. **You have the legal right** to share every byte you submit.
2. You will **respect privacy**: redact bystander faces, license plates, and
   personally identifiable information (PII) unless the data subject has
   given informed, written consent.
3. You will **attribute sources** accurately. Plagiarism, falsified provenance,
   and AI-fabricated "sightings" are project-banning offenses.
4. You will **not weaponise** the dataset for harassment, doxxing, or
   targeting individuals (witnesses, scientists, or military personnel).
5. You will **not** circumvent national laws — including FOIA-exemption
   handling, GDPR, and equivalents — to obtain data for submission.
6. You consent to maintainers **removing** your contribution if a legal or
   ethical issue is discovered post-merge, with attribution preserved in the
   git history per the project license.

Violations result in submission rejection, repository ban, and — for
suspected criminal violations — referral to the appropriate authorities.

---

## 5. Jurisdictional Notes

- **United States**: ITAR, EAR, Atomic Energy Act, Espionage Act,
  18 U.S.C. § 798 (COMINT).
- **European Union**: Regulation (EU) 2021/821 (dual-use), GDPR
  (Regulation (EU) 2016/679).
- **United Kingdom**: Official Secrets Acts 1911–1989, Export Control Act 2002.
- **Australia / Canada / NZ**: Five Eyes equivalents apply.
- **France** (relevant to GEIPAN data): Code de la défense L. 2311-1 et seq.
- **Chile** (relevant to CEFAA data): Ley 19.974 (intelligence) and
  DAN-91 (military airspace).

The project hosts only material lawfully releasable in **all** of these
jurisdictions.

---

## 6. Rejection Template

```
Hello @contributor,

Thank you for your submission. Unfortunately we cannot accept this
contribution because it appears to contain material that may be:

  - [ ] classified or formerly classified (banner/portion marks observed)
  - [ ] ITAR/EAR-controlled (USML/CCL category referenced)
  - [ ] obtained without legal authority
  - [ ] containing PII without consent

Please review docs/LEGAL_COMPLIANCE.md and, if you believe this triage was in
error, respond with the formal declassification / public-release reference.

This issue will remain closed pending such a response.
```

---

## 7. Legal Disclaimer

Nothing in this document constitutes legal advice. The project is provided
"as-is" without warranty. Contributors are solely responsible for their
submissions and should consult qualified counsel when in doubt.

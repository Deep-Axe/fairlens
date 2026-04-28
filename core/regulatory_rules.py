REGULATORY_TEXT = {
    "EEOC_4_5THS": (
        'EEOC Uniform Guidelines on Employee Selection Procedures, 29 CFR Part 1607, Section 4(D): '
        '"A selection rate for any race, sex, or ethnic group which is less than four-fifths (4/5) '
        'of the rate for the group with the highest rate will generally be regarded by Federal '
        'enforcement agencies as evidence of adverse impact."'
    ),
    "EU_AI_ACT": (
        "EU Artificial Intelligence Act (2024), Annex III, Point 2: AI systems intended to be used "
        "for recruitment or selection of natural persons, including screening or filtering applications "
        "and evaluating candidates, are classified as high-risk. "
        "Article 10(2)(f): Training, validation and testing data sets shall be subject to data "
        "governance practices concerning examination in view of possible biases that are likely to "
        "affect health and safety of persons or lead to discrimination prohibited by Union law."
    ),
    "FCRA_615": (
        "Fair Credit Reporting Act, 15 U.S.C. § 1681m (Section 615): If any person takes any adverse "
        "action with respect to any consumer that is based in whole or in part on any information "
        "contained in a consumer report, the person shall provide notice to the consumer of the "
        "adverse action, the name and address of the consumer reporting agency, and the consumer's "
        "right to obtain a free copy of the report and to dispute its accuracy."
    ),
    "ECOA": (
        "Equal Credit Opportunity Act (Regulation B), 15 U.S.C. § 1691, 12 CFR Part 1002: "
        "It shall be unlawful for any creditor to discriminate against any applicant, with respect "
        "to any aspect of a credit transaction, on the basis of race, color, religion, national "
        "origin, sex or marital status, or age."
    ),
}


def evaluate_regulatory_compliance(audit_results):
    """
    Deterministically maps audit metric outcomes to regulatory flags.
    No LLM — pure threshold checks against published regulatory standards.
    Returns a list of triggered regulation dicts with severity and required actions.
    """
    flags = []

    layer1 = audit_results.get("layer1", {})
    layer2 = audit_results.get("layer2", {})
    layer3 = audit_results.get("layer3", {})

    di_scores = layer1.get("disparate_impact_data", {})
    fairness_gaps = layer2.get("fairness_gaps", {})
    probe_accuracies = layer3.get("probe_accuracies", {})
    counterfactual_flips = layer2.get("counterfactual_flips", {})

    # EEOC 4/5ths Rule — disparate impact below 0.80
    for attr, di in di_scores.items():
        if di is not None and di < 0.80:
            flags.append({
                "rule": "EEOC_4_5THS",
                "severity": "VIOLATION",
                "regulation": "EEOC Uniform Guidelines on Employee Selection Procedures",
                "citation": "29 CFR Part 1607, Section 4(D)",
                "finding": f"Disparate impact ({attr}): {di:.2f} — below the 0.80 threshold",
                "required_action": (
                    "Demonstrate job-relatedness and business necessity, or discontinue use "
                    "in employment selection. Document analysis before next deployment."
                ),
                "regulatory_text": REGULATORY_TEXT["EEOC_4_5THS"],
            })

    # EU AI Act — protected attribute reconstructible from model internals above 70%
    for attr, layers in probe_accuracies.items():
        max_acc = max(layers.values()) if layers else 0
        if max_acc > 0.70:
            flags.append({
                "rule": "EU_AI_ACT",
                "severity": "COMPLIANCE_REVIEW_REQUIRED",
                "regulation": "EU Artificial Intelligence Act",
                "citation": "Annex III (High-Risk Systems) + Article 10 (Data Governance)",
                "finding": (
                    f"Protected attribute '{attr}' detectable from model internals at "
                    f"{max_acc:.0%} accuracy — model has reconstructed the protected "
                    "attribute from proxy features despite it being excluded from inputs"
                ),
                "required_action": (
                    "Conduct Fundamental Rights Impact Assessment before EU deployment. "
                    "Implement data governance review per Article 10. "
                    "Apply surgical debiasing (Layer 4) and re-audit."
                ),
                "regulatory_text": REGULATORY_TEXT["EU_AI_ACT"],
            })
            break  # one EU AI Act flag per audit is sufficient

    # FCRA Section 615 — counterfactual sensitivity detected
    total_flips = sum(counterfactual_flips.values()) if counterfactual_flips else 0
    if total_flips > 0:
        flags.append({
            "rule": "FCRA_615",
            "severity": "NOTICE_REQUIRED_IF_CREDIT",
            "regulation": "Fair Credit Reporting Act",
            "citation": "15 U.S.C. § 1681m (Section 615)",
            "finding": (
                f"{total_flips} counterfactual flips detected — model decisions change "
                "when only the protected attribute is altered with all else equal"
            ),
            "required_action": (
                "If used in any credit, lending, or insurance decision: provide adverse "
                "action notices to denied applicants specifying the reasons for denial."
            ),
            "regulatory_text": REGULATORY_TEXT["FCRA_615"],
        })

    # ECOA / Regulation B — demographic parity difference above 0.10 in credit context
    for attr, gaps in fairness_gaps.items():
        dpd = abs(gaps.get("demographic_parity_diff", 0))
        if dpd > 0.10:
            flags.append({
                "rule": "ECOA",
                "severity": "INVESTIGATION_WARRANTED",
                "regulation": "Equal Credit Opportunity Act (Regulation B)",
                "citation": "15 U.S.C. § 1691, 12 CFR Part 1002",
                "finding": (
                    f"Demographic parity difference ({attr}): {dpd:.2f} — "
                    "selection rates differ significantly across protected groups"
                ),
                "required_action": (
                    "If used in credit or lending: conduct formal disparate impact analysis. "
                    "Document business necessity justification or implement remediation."
                ),
                "regulatory_text": REGULATORY_TEXT["ECOA"],
            })

    return flags

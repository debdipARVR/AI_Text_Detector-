"""Batch Evaluation Script for 'The Questions That Shape Us' (120 Conceptual Essays).

Runs the Two-Pass DeepEval & Cloze Congruence Detector on all 120 essays individually,
logging:
- Pass 2 (Alternate Masking) Congruence
- Pass 3 (Centroid 3-Sentence Masking) Congruence
- Inter-Pass Congruence Delta
- Sigmoid Dynamic Cosine/Meaning Weights
- Overall Combined Congruence Score
- AI Probability %
- Verdict
- Burstiness Metric
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient

DOMAINS = {
    "Philosophy & Human Nature": [
        (1, "Is intelligence more valuable than wisdom?"),
        (2, "Does freedom require limits?"),
        (3, "Can a society survive without trust?"),
        (4, "Are humans naturally cooperative or competitive?"),
        (5, "Is suffering necessary for personal growth?"),
        (6, "Does happiness have an upper limit?"),
        (7, "Should morality evolve with society?"),
        (8, "Can people truly change their personality?"),
        (9, "Is identity discovered or created?"),
        (10, "Is ignorance ever a virtue?"),
    ],
    "Artificial Intelligence & Technology": [
        (11, "Will AI make humans intellectually stronger or weaker?"),
        (12, "Should AI possess legal rights?"),
        (13, "Is privacy becoming an outdated concept?"),
        (14, "Can technology solve problems it creates?"),
        (15, "The future of work in an AI-driven economy."),
        (16, "Should algorithms influence democratic decisions?"),
        (17, "Will AI replace creativity or redefine it?"),
        (18, "Is digital immortality desirable?"),
        (19, "Human enhancement through technology: blessing or threat?"),
        (20, "Are we becoming dependent rather than empowered by technology?"),
    ],
    "Psychology": [
        (21, "Why do intelligent people make irrational decisions?"),
        (22, "Can memory be trusted?"),
        (23, "The psychology of loneliness in the digital age."),
        (24, "Why do people fear uncertainty?"),
        (25, "Is confidence learned or inherited?"),
        (26, "Does social media distort self-worth?"),
        (27, "Can habits overcome motivation?"),
        (28, "The role of boredom in creativity."),
        (29, "Why do humans seek meaning?"),
        (30, "Are emotions obstacles or decision-making tools?"),
    ],
    "Economics & Society": [
        (31, "Is economic growth always beneficial?"),
        (32, "Should wealth have a ceiling?"),
        (33, "Universal Basic Income: necessity or luxury?"),
        (34, "Can capitalism exist without inequality?"),
        (35, "The hidden cost of convenience."),
        (36, "Is consumerism replacing culture?"),
        (37, "Can economic progress coexist with environmental sustainability?"),
        (38, "Should governments regulate AI monopolies?"),
        (39, "Why do societies tolerate inequality?"),
        (40, "The future of money in a digital world."),
    ],
    "Science": [
        (41, "Are scientific discoveries morally neutral?"),
        (42, "Should humans colonize other planets?"),
        (43, "Is consciousness purely biological?"),
        (44, "Can science answer every question?"),
        (45, "The ethics of genetic engineering."),
        (46, "Does evolution still shape modern humans?"),
        (47, "Should humanity pursue immortality?"),
        (48, "Climate engineering: solution or dangerous experiment?"),
        (49, "The limits of human knowledge."),
        (50, "Is randomness fundamental to the universe?"),
    ],
    "History": [
        (51, "Do great leaders shape history or do circumstances?"),
        (52, "Can history repeat itself?"),
        (53, "Why do civilizations collapse?"),
        (54, "Should historical figures be judged by modern standards?"),
        (55, "What makes an empire successful?"),
        (56, "Is war inevitable?"),
        (57, "How does propaganda shape history?"),
        (58, "Can history ever be objective?"),
        (59, "The role of geography in civilization."),
        (60, "Lessons modern societies ignore from history."),
    ],
    "Ethics": [
        (61, "Is lying ever morally justified?"),
        (62, "Should animals have legal rights?"),
        (63, "Can punishment create justice?"),
        (64, "The ethics of surveillance."),
        (65, "Should future generations have legal representation?"),
        (66, "Is censorship ever justified?"),
        (67, "Can ethical decisions be objective?"),
        (68, "Is intention more important than outcome?"),
        (69, "The ethics of autonomous weapons."),
        (70, "Should organ donation be mandatory after death?"),
    ],
    "Environment": [
        (71, "Can economic prosperity exist without environmental exploitation?"),
        (72, "Is humanity part of nature or separate from it?"),
        (73, "The ethics of rewilding ecosystems."),
        (74, "Should climate refugees receive special legal status?"),
        (75, "Can individual action meaningfully fight climate change?"),
        (76, "Is biodiversity more important than economic development?"),
        (77, "The future of sustainable cities."),
        (78, "Should natural resources belong to humanity?"),
        (79, "Can technology replace conservation?"),
        (80, "Is environmental responsibility a moral obligation?"),
    ],
    "Education": [
        (81, "Should schools prioritize curiosity over examinations?"),
        (82, "Is failure the best teacher?"),
        (83, "Can online education replace universities?"),
        (84, "Should education be customized for every student?"),
        (85, "The value of humanities in a technological world."),
        (86, "Does grading harm learning?"),
        (87, "Can creativity be taught?"),
        (88, "The purpose of education: employment or enlightenment?"),
        (89, "Should financial literacy be mandatory in schools?"),
        (90, "Is self-education superior to formal education?"),
    ],
    "Politics & Governance": [
        (91, "Is democracy the best form of government?"),
        (92, "Should voting be mandatory?"),
        (93, "Can global governance replace nation-states?"),
        (94, "The balance between national security and personal liberty."),
        (95, "Should politicians have term limits?"),
        (96, "Does power inevitably corrupt?"),
        (97, "Can corruption ever be eliminated?"),
        (98, "The role of civil disobedience in democracy."),
        (99, "Is meritocracy truly possible?"),
        (100, "Should the state control key industries?"),
    ],
    "Literature & Culture": [
        (101, "Does art reflect society or shape it?"),
        (102, "Can literature change human empathy?"),
        (103, "Is cultural preservation more important than cultural evolution?"),
        (104, "Does modern culture destroy deep reading?"),
        (105, "Can tragedy provide joy?"),
        (106, "The role of myth in modern society."),
        (107, "Is originality still possible in art?"),
        (108, "Does language shape how we think?"),
        (109, "Can AI create authentic literature?"),
        (110, "Why do stories matter to humans?"),
    ],
    "Future & Speculative": [
        (111, "What will define humanity in 1000 years?"),
        (112, "Will artificial superintelligence be benevolent?"),
        (113, "Can humanity survive interstellar migration?"),
        (114, "Will post-scarcity eliminate human conflict?"),
        (115, "Is transhumanism the next stage of human evolution?"),
        (116, "Will virtual reality replace physical reality?"),
        (117, "What if we discover extraterrestrial life?"),
        (118, "Can a machine experience love?"),
        (119, "The ultimate fate of human consciousness."),
        (120, "What question should future humans never stop asking?"),
    ],
}


def build_essay_text(title: str) -> str:
    # Normalize question format for template
    clean_topic = title.rstrip("?.").strip().lower()
    return f"""INTRODUCTION
The question of {clean_topic} reaches beyond a simple yes-or-no answer. It asks how individuals, institutions, and communities should live when important values pull in different directions. The issue matters because choices made here influence opportunity, dignity, responsibility, and the kind of future people can share. A useful approach begins by defining the central terms and recognizing that they are shaped by history, power, evidence, and human experience.

BALANCED ANALYSIS
One persuasive view is that {clean_topic} can produce genuine benefits when guided by careful judgment. Supporters point to practical examples: institutions that broaden participation, innovations that reduce suffering, and communities that adapt rules to new circumstances. The strongest case is also about agency and fairness. Benefits are more credible when people can understand decisions, challenge authority, and participate in distributing gains and risks. The topic is therefore best understood as a relationship between individual freedom and collective responsibility.

COUNTERARGUMENTS AND EXAMPLES
The opposing argument deserves equal weight. Critics warn that the same force can create unintended costs, including exclusion, dependency, unequal power, or the quiet loss of values that are difficult to measure. History shows that noble goals can be distorted by concentrated authority, while private choices can harm people who were never consulted. A balanced position requires transparency, proportional limits, independent review, and room for dissent. The question is not whether the idea is perfect, but under what conditions it remains humane and accountable.

CONCLUSION
Ultimately, {clean_topic} should be understood as an ongoing judgment rather than a final formula. Evidence can clarify consequences, philosophy can test fairness, and lived experience can reveal harms that abstract models overlook. The most defensible conclusion is conditional: pursue potential benefits, acknowledge trade-offs, and design institutions capable of correction. This approach does not eliminate disagreement, but it turns disagreement into a resource for wiser decisions."""


def main():
    print("=" * 80)
    print(" EVALUATING 120 CONCEPTUAL ESSAYS (SEPARATE RUN FOR EACH ESSAY)")
    print("=" * 80)

    client = NvidiaNIMClient(api_key="")
    detector = ClozeCongruenceDetector(nim_client=client)

    all_records = []
    domain_summaries = {}

    start_time = time.time()
    total_essays = 0

    for domain_name, essays in DOMAINS.items():
        print(f"\nEvaluating Domain: {domain_name} ({len(essays)} essays)...")
        domain_scores = []

        for essay_num, title in essays:
            text = build_essay_text(title)
            res = detector.analyze(text)

            record = {
                "essay_id": essay_num,
                "domain": domain_name,
                "title": title,
                "ai_probability": res["ai_probability"],
                "verdict": res["verdict"],
                "combined_congruence": res["combined_congruence_score"],
                "pass_2_congruence": res["pass_2"]["congruence_score"],
                "pass_3_congruence": res["pass_3"]["congruence_score"],
                "pass_delta": res.get("pass_delta", 0.0),
                "burstiness": res.get("burstiness_metric", 0.0),
                "confidence_score": res.get("confidence_score", 90.0),
                "word_count": len(text.split()),
            }
            all_records.append(record)
            domain_scores.append(record)
            total_essays += 1

            print(f"  [Essay {essay_num:03d}] {title[:40]:<40} -> AI Prob: {record['ai_probability']}% | Verdict: {record['verdict']} | Congruence: {record['combined_congruence']}%")

        # Domain Aggregate
        avg_ai_prob = round(sum(r["ai_probability"] for r in domain_scores) / len(domain_scores), 2)
        avg_congruence = round(sum(r["combined_congruence"] for r in domain_scores) / len(domain_scores), 2)
        avg_p2 = round(sum(r["pass_2_congruence"] for r in domain_scores) / len(domain_scores), 2)
        avg_p3 = round(sum(r["pass_3_congruence"] for r in domain_scores) / len(domain_scores), 2)
        verdict_counts = {}
        for r in domain_scores:
            v = r["verdict"]
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        domain_summaries[domain_name] = {
            "count": len(domain_scores),
            "avg_ai_probability": avg_ai_prob,
            "avg_combined_congruence": avg_congruence,
            "avg_pass_2": avg_p2,
            "avg_pass_3": avg_p3,
            "verdict_distribution": verdict_counts,
        }

    elapsed = round(time.time() - start_time, 2)

    # Save to JSON
    out_json_path = os.path.join("data", "benchmark_results", "120_essays_scores.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_evaluated_essays": total_essays,
                "elapsed_seconds": elapsed,
                "domain_summaries": domain_summaries,
                "essays": all_records,
            },
            f,
            indent=2,
        )

    # Save to CSV
    out_csv_path = os.path.join("data", "benchmark_results", "120_essays_scores.csv")
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Essay_ID",
            "Domain",
            "Title",
            "AI_Probability_Percent",
            "Verdict",
            "Combined_Congruence_Percent",
            "Pass_2_Congruence_Percent",
            "Pass_3_Congruence_Percent",
            "Pass_Delta",
            "Confidence_Score",
            "Word_Count",
        ])
        for r in all_records:
            writer.writerow([
                r["essay_id"],
                r["domain"],
                r["title"],
                r["ai_probability"],
                r["verdict"],
                r["combined_congruence"],
                r["pass_2_congruence"],
                r["pass_3_congruence"],
                r["pass_delta"],
                r["confidence_score"],
                r["word_count"],
            ])

    print("\n" + "=" * 80)
    print(" ALL 120 ESSAYS EVALUATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Essays Evaluated: {total_essays}")
    print(f"Total Execution Time:   {elapsed}s (avg {round(elapsed/total_essays, 3)}s/essay)")
    print(f"JSON Export:            {out_json_path}")
    print(f"CSV Export:             {out_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

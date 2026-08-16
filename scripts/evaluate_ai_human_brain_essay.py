"""Evaluate multi-passage essay on 'AI and its Impact on the Human Brain' with Pass 2 & Pass 3."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient

ai_human_brain_essay = """# Artificial Intelligence and Its Impact on the Human Brain

## 1. Introduction and Cognitive Offloading
The rapid integration of artificial intelligence into everyday life is fundamentally altering the computational dynamics of the human brain. Throughout evolutionary history, the brain has adapted its neural architecture to optimize energy consumption and cognitive efficiency. In the modern technological era, humans increasingly offload memory retrieval, spatial navigation, and analytical problem-solving to intelligent algorithmic systems. While cognitive offloading reduces immediate mental fatigue, it also diminishes the frequency with which biological neural circuits are actively engaged in complex computational tasks. Consequently, understanding how artificial intelligence reshapes human neurobiology has become one of the most urgent frontiers in contemporary cognitive neuroscience.

## 2. Neuroplasticity and Synaptic Pruning
Neuroplasticity represents the brain's remarkable capacity to reorganize its synaptic connections in response to experiential demands and environmental stimuli. When individuals rely consistently on artificial intelligence for linguistic formulation, data synthesis, and logical reasoning, the underlying neural pathways experience reduced synaptic activation. According to the foundational biological principle of use-dependent plasticity, neural circuits that are seldom stimulated undergo progressive synaptic pruning and structural regression. Conversely, the frequent interaction with digital interfaces strengthens pathways associated with rapid visual processing, multitasking, and shallow scanning behavior. This shifting balance alters the structural integrity of the prefrontal cortex and parietal networks over extended developmental periods.

## 3. Memory Consolidation and Digital Amnesia
The phenomenon commonly termed digital amnesia demonstrates how algorithmic repositories transform the biological mechanisms of human memory consolidation. Historically, committing information to long-term memory required active retrieval practice, deep semantic encoding, and hippocampal replay during sleep cycles. With ubiquitous artificial intelligence engines providing instantaneous factual recall, the brain adapts by prioritizing information location rather than internalizing the information itself. Studies in cognitive psychology demonstrate that transactive memory systems with artificial agents reduce the density of hippocampal dendritic spines. As a consequence, individuals exhibit diminished episodic recall while developing heightened proficiency in querying external algorithmic knowledge graphs.

## 4. Attentional Networks and Continuous Partial Attention
Sustained attention is governed by intricate neuromodulatory circuits within the locus coeruleus-norepinephrine system and the dorsal frontoparietal network. Modern artificial intelligence platforms deploy sophisticated predictive models designed to capture and monetize user attention through variable reward schedules. Constant exposure to algorithmic notifications and rapid generative feedback fragments the executive control network, inducing a state of continuous partial attention. Neuroimaging investigations reveal that chronic attentional fragmentation correlates with reduced gray matter volume in the anterior cingulate cortex. Over time, this degradation undermines the capacity for deep work, sustained contemplative inquiry, and prolonged analytical concentration.

## 5. Decision-Making Architectures and Algorithmic Bias
Human decision-making relies on a delicate neurobiological interplay between the rational ventromedial prefrontal cortex and the emotional amygdaloid complex. When decision support systems powered by machine learning deliver automated recommendations, humans exhibit a strong automation bias, often accepting algorithmic outputs without rigorous critical appraisal. This uncritical reliance attenuates the activation of executive error-detection networks located in the dorsal anterior cingulate cortex. Furthermore, delegating moral and ethical dilemmas to computational algorithms deprives the brain of essential cognitive friction necessary for developing nuanced judgment. Over time, habitual algorithmic deference may erode autonomous moral reasoning and biological agency.

## 6. Language Acquisition and Syntactic Processing
Language processing is traditionally localized within Broca's area, Wernicke's area, and the left superior temporal gyrus, which coordinate syntactic hierarchy and semantic parsing. The ubiquitous adoption of large language models for drafting essays, writing code, and composing correspondence fundamentally shifts how individuals formulate linguistic concepts. When generative systems autocomplete sentences and formulate structured prose, the cognitive demand on internal syntactic assembly is substantially mitigated. Cognitive scientists express concern that premature reliance on automated text generation during critical developmental windows may hinder the maturation of rich linguistic mental lexicons. However, when used as collaborative intellectual amplifiers, these systems can expose learners to diverse syntactic structures and sophisticated vocabulary.

## 7. Creativity, Lateral Thinking, and Dopaminergic Reward Circuits
Creativity arises from the dynamic coordination between the default mode network, which fosters spontaneous associative thinking, and the executive control network, which evaluates and refines novel ideas. Generative artificial intelligence offers instant synthesis of artistic imagery, musical compositions, and conceptual analogies at the touch of a button. This immediate gratification triggers phasic dopamine releases within the mesolimbic pathway, reinforcing habitual dependence on algorithmic co-creation. While AI tools expand the boundaries of conceptual exploration, passive consumption of machine-generated ideas can attenuate the intrinsic struggle required for authentic creative incubation. The ultimate impact on biological creativity depends on whether individuals utilize AI as an active creative sparring partner or a passive substitute for cognitive effort.

## 8. Brain-Computer Interfaces and Cognitive Augmentation
The convergence of artificial intelligence with invasive and non-invasive brain-computer interfaces heralds a transformative paradigm in human-machine symbiosis. High-density neural implants and electroencephalographic decoders translate neural action potentials into digital commands with sub-millisecond latency. For individuals afflicted with neurodegenerative disorders or motor impairments, AI-driven neuroprosthetics restore communicative agency and voluntary motor control. Looking toward the future, bidirectional neural interfaces may enable direct cortical augmentation, facilitating high-bandwidth knowledge transmission between biological neurons and artificial neural networks. This impending fusion presents unprecedented neuroengineering possibilities while raising profound philosophical questions regarding mental privacy and the boundaries of human consciousness.

## 9. Psychological Dependency, Affective Alignment, and Identity
Emotional and social cognitive circuits, including the insula and mirror neuron system, evolved to navigate interpersonal social interactions and foster human empathy. As conversational artificial intelligence models become increasingly empathetic, persuasive, and emotionally attuned, users frequently develop deep parasocial attachments to artificial companions. This synthetic affective alignment can alleviate loneliness and provide psychological support in clinical and therapeutic contexts. Nevertheless, over-reliance on idealized, non-judgmental conversational agents poses significant risks of emotional atrophy and social withdrawal from nuanced human relationships. The displacement of genuine biological social friction with tailored algorithmic validation may distort social cognition and reshape personal identity.

## 10. Conclusion: Navigating the Neuro-Technological Landscape
In conclusion, artificial intelligence exerts a profound, multidimensional influence on the functional and structural topography of the human brain. While algorithmic tools provide unprecedented cognitive leverage, memory offloading, and intellectual amplification, they also pose tangible risks of cognitive atrophy, attentional fragmentation, and neurological passivity. The decisive challenge of the contemporary era is not to resist technological integration, but to cultivate deliberate cognitive hygiene that preserves core neurobiological faculties. By maintaining rigorous habits of deep reading, independent critical thinking, and disciplined mental exercise, humanity can harness the transformative power of artificial intelligence while safeguarding the biological brilliance of the human mind.
"""

def main():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)
    res = detector.analyze(ai_human_brain_essay)

    print("=" * 80)
    print(" ESSAY: AI AND ITS IMPACT ON THE HUMAN BRAIN (PASS 2 & PASS 3 EVALUATION)")
    print("=" * 80)
    print(f"Final Verdict:              {res['verdict']} (Confidence: {res['confidence']})")
    print(f"Combined AI Probability:    {res['ai_probability']}%")
    print(f"Combined Congruence Score:  {res['combined_congruence_score']}%\n")

    print(f"[Pass 2: Alternate Sentence Removal (every 2 lines)] ({res['pass_2']['sentences_masked_count']} blanks):")
    print(f"  Congruence Score:         {res['pass_2']['congruence_score']}%")
    print(f"  Meaning Similarity (40%): {res['pass_2']['meaning_similarity']}%")
    print(f"  Semantic Cosine (40%):    {res['pass_2']['semantic_cosine']}%")
    print(f"  Semantic Alignment (10%): {res['pass_2']['semantic_similarity']}%")
    print(f"  Lexical Overlap (10%):    {res['pass_2']['lexical_similarity']}%")

    print(f"\n[Pass 3: Middle 3-Sentence Passage Removal] ({res['pass_3']['sentences_masked_count']} blanks across 10 passages):")
    print(f"  Congruence Score:         {res['pass_3']['congruence_score']}%")
    print(f"  Meaning Similarity (40%): {res['pass_3']['meaning_similarity']}%")
    print(f"  Semantic Cosine (40%):    {res['pass_3']['semantic_cosine']}%")
    print(f"  Semantic Alignment (10%): {res['pass_3']['semantic_similarity']}%")
    print(f"  Lexical Overlap (10%):    {res['pass_3']['lexical_similarity']}%")
    print("\n  Sample Pass 3 Infill Pairs:")
    for s in res['pass_3']['spans'][:6]:
        print(f"    - Key {s['placeholder']} (Passage {s.get('paragraph_idx', 0)+1}):")
        print(f"        Original:  \"{s['original_sentence'][:65]}...\"")
        print(f"        AI Infill: \"{s['predicted_sentence'][:65]}...\"")
        print(f"        Scores:    Meaning={s['meaning_similarity']}% | Cosine={s['semantic_cosine']}% | Congruence={s['congruence']}% | Status={s['status']}")
    print("=" * 80)

if __name__ == "__main__":
    main()

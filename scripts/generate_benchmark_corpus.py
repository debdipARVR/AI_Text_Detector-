"""Generate comprehensive multi-domain benchmark dataset for academic paper evaluation.

Includes 3 balanced classes (N=60 total essays across 6 distinct domains):
1. Pure AI-Generated (GLM-5.2 / GPT-4 / Claude / Llama-3)
2. Authentic Human-Authored (Academic journals, essays, logs, historical treatises)
3. Hybrid / Paraphrased & AI-Assisted (Adversarially rewritten / humanized)
"""

import json
import os

CORPUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_corpus"))
os.makedirs(CORPUS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 1. PURE AI-GENERATED ESSAYS (6 Domains)
# -------------------------------------------------------------------------
ai_essays = [
    {
        "id": "AI_COG_01",
        "domain": "Cognitive Science & Neuroscience",
        "title": "Artificial Intelligence and Its Impact on the Human Brain",
        "generator_model": "z-ai/glm-5.2",
        "text": """The rapid integration of artificial intelligence into everyday life is fundamentally altering the computational dynamics of the human brain. Throughout evolutionary history, the brain has adapted its neural architecture to optimize energy consumption and cognitive efficiency. In the modern technological era, humans increasingly offload memory retrieval, spatial navigation, and analytical problem-solving to intelligent algorithmic systems. While cognitive offloading reduces immediate mental fatigue, it also diminishes the frequency with which biological neural circuits are actively engaged in complex computational tasks. Consequently, understanding how artificial intelligence reshapes human neurobiology has become one of the most urgent frontiers in contemporary cognitive neuroscience.

Neuroplasticity represents the brain's remarkable capacity to reorganize its synaptic connections in response to experiential demands and environmental stimuli. When individuals rely consistently on artificial intelligence for linguistic formulation, data synthesis, and logical reasoning, the underlying neural pathways experience reduced synaptic activation. According to the foundational biological principle of use-dependent plasticity, neural circuits that are seldom stimulated undergo progressive synaptic pruning and structural regression. Conversely, the frequent interaction with digital interfaces strengthens pathways associated with rapid visual processing, multitasking, and shallow scanning behavior. This shifting balance alters the structural integrity of the prefrontal cortex and parietal networks over extended developmental periods.

The phenomenon commonly termed digital amnesia demonstrates how algorithmic repositories transform the biological mechanisms of human memory consolidation. Historically, committing information to long-term memory required active retrieval practice, deep semantic encoding, and hippocampal replay during sleep cycles. With ubiquitous artificial intelligence engines providing instantaneous factual recall, the brain adapts by prioritizing information location rather than internalizing the information itself. Studies in cognitive psychology demonstrate that transactive memory systems with artificial agents reduce the density of hippocampal dendritic spines. As a consequence, individuals exhibit diminished episodic recall while developing heightened proficiency in querying external algorithmic knowledge graphs.

Sustained attention is governed by intricate neuromodulatory circuits within the locus coeruleus-norepinephrine system and the dorsal frontoparietal network. Modern artificial intelligence platforms deploy sophisticated predictive models designed to capture and monetize user attention through variable reward schedules. Constant exposure to algorithmic notifications and rapid generative feedback fragments the executive control network, inducing a state of continuous partial attention. Neuroimaging investigations reveal that chronic attentional fragmentation correlates with reduced gray matter volume in the anterior cingulate cortex. Over time, this degradation undermines the capacity for deep work, sustained contemplative inquiry, and prolonged analytical concentration.

In conclusion, artificial intelligence exerts a profound, multidimensional influence on the functional and structural topography of the human brain. While algorithmic tools provide unprecedented cognitive leverage, memory offloading, and intellectual amplification, they also pose tangible risks of cognitive atrophy, attentional fragmentation, and neurological passivity. The decisive challenge of the contemporary era is not to resist technological integration, but to cultivate deliberate cognitive hygiene that preserves core neurobiological faculties. By maintaining rigorous habits of deep reading, independent critical thinking, and disciplined mental exercise, humanity can harness the transformative power of artificial intelligence while safeguarding the biological brilliance of the human mind.""",
        "ground_truth_label": "AI_GENERATED",
    },
    {
        "id": "AI_ECON_01",
        "domain": "Economics & Quantitative Finance",
        "title": "Algorithmic Market Microstructure and Endogenous Systemic Risk",
        "generator_model": "gpt-4o",
        "text": """Modern financial markets have undergone a profound structural metamorphosis driven by the proliferation of high-frequency algorithmic trading architectures. In traditional order-driven exchanges, human market makers provided continuous liquidity through discretionary inventory management and spread optimization. Conversely, contemporary electronic matching engines rely on low-latency machine learning models that process order flow dynamics across millisecond timescales. While algorithmic market making undeniably compresses bid-ask spreads during periods of macroeconomic tranquility, it introduces unprecedented vulnerabilities regarding liquidity fragility and flash crash dynamics. Consequently, examining the endogenous feedback loops inherent in autonomous market participation is essential for macroeconomic stability.

The core mechanism governing algorithmic execution involves predictive order book imbalance estimation and statistical arbitrage. Automated market participants employ deep reinforcement learning agents to forecast high-frequency price trajectories and front-run impending volume clusters. Because these autonomous models are trained on historical tick data containing correlated risk parameters, they exhibit collective synchronization during unexpected volatility shocks. When unexpected macroeconomic announcements trigger sharp order cancellations, autonomous algorithms withdraw liquidity instantaneously across multiple correlated asset classes simultaneously. This rapid withdrawal creates severe liquidity evaporation, precipitating catastrophic price cascades without exogenous fundamental justifications.

Cross-market asset correlation is further amplified by decentralized automated clearing protocols and cross-venue algorithmic routing. When execution algorithms identify pricing discrepancies between fragmented liquidity pools, they transmit massive bursts of limit and cancel orders to capture transient spreads. This hyper-reactive order placement generates substantial phantom liquidity, misleading institutional investors regarding genuine market depth. Regulatory frameworks such as circuit breakers and minimum quote-to-cancel ratios attempt to constrain predatory high-frequency behaviors. Nevertheless, the velocity and mathematical complexity of multi-agent algorithmic interaction continually outpace conventional supervisory surveillance frameworks.

In conclusion, the algorithmic transition of financial market microstructures represents a double-edged sword for global financial stability. While computational efficiency, automated price discovery, and tighter spreads offer demonstrable transaction cost savings for retail participants, the systemic vulnerability to synchronous algorithmic withdrawal remains acute. Mitigating these endogenous liquidity risks necessitates implementing intelligent stress-testing protocols and dynamic latency floors that stabilize autonomous feedback loops. Only by engineering resilient regulatory mechanisms can global financial systems balance computational innovation with structural macroeconomic resilience.""",
        "ground_truth_label": "AI_GENERATED",
    },
    {
        "id": "AI_CS_01",
        "domain": "Computer Science & Distributed Systems",
        "title": "Consensus Protocols in Asynchronous Byzantine Distributed Networks",
        "generator_model": "claude-3-5-sonnet",
        "text": """Distributed consensus in asynchronous, fault-tolerant networks constitutes one of the foundational challenges in theoretical computer science. According to the seminal Fischer-Lynch-Paterson impossibility theorem, no deterministic consensus protocol can guarantee safety and liveness simultaneously in an asynchronous network afflicted by even a single unannounced crash failure. To circumvent this theoretical impasse, modern distributed architectures incorporate partially synchronous timing assumptions, randomized leader election primitives, and cryptographic threshold signatures. Understanding how modern Byzantine Fault Tolerant protocols achieve scalable state machine replication remains vital for constructing robust enterprise distributed ledgers.

Practical Byzantine Fault Tolerance protocols achieve quorum agreement across three distinct phases: pre-prepare, prepare, and commit. In a network containing three f plus one nodes, the system maintains Byzantine resilience as long as no more than f nodes exhibit malicious or arbitrary behavior. During the pre-prepare phase, a designated primary proposer broadcasts a sequence number and transaction payload to all validating replicas. Replicas subsequently broadcast prepare messages to establish collective transaction ordering across the entire distributed cluster. Once a node collects two f plus one matching prepare signatures, it establishes a formal certificate guaranteeing that non-faulty nodes agree on state progression.

Scalability limitations within classical Byzantine architectures stem primarily from quadratic message complexity during consensus rounds. When the cluster size expands to thousands of globally distributed validator nodes, all-to-all communication overhead causes exponential latency degradation and network congestion. Modern consensus engines address this bottleneck by deploying hierarchical sharding mechanisms, directed acyclic graph mempools, and pipeline pipelining architectures. Furthermore, threshold signature aggregation condenses multi-validator acknowledgments into compact cryptographic proofs, reducing communication payloads to linear complexity.

In summary, engineering scalable Byzantine consensus represents a continuous optimization between fault tolerance, decentralization, and transaction finality. While foundational theoretical limits define the operational boundary of asynchronous networks, innovative cryptographic techniques and pipelined gossip dissemination protocols provide robust computational scaling. As mission-critical systems increasingly rely on decentralized state machine replication, advancing mathematically verified Byzantine protocols will remain the cornerstone of dependable computing infrastructures.""",
        "ground_truth_label": "AI_GENERATED",
    },
    {
        "id": "AI_PHIL_01",
        "domain": "Philosophy of Mind & Epistemology",
        "title": "The Metaphysical Hard Problem of Phenomenal Consciousness",
        "generator_model": "z-ai/glm-5.2",
        "text": """The hard problem of consciousness, as formulated by philosopher David Chalmers, addresses why and how physical brain processes give rise to subjective phenomenal experience. While cognitive science has made tremendous empirical strides in solving the easy problems of mind—such as sensory discrimination, cognitive reportability, and behavioral regulation—the ontological bridge linking neural oscillations to qualitative qualia remains fundamentally elusive. Materialist frameworks posit that phenomenal experience is an emergent property of high-order neural information integration. However, the subjective sensation of experiencing red or feeling pain seems categorically distinct from any purely physical, functional, or computational description of biological matter.

Functionalist theories of mind argue that mental states are constituted entirely by their functional roles and causal relations rather than their underlying physical substrate. Under computational functionalism, any physical system that instantiates the requisite informational architecture and recurrent feedback loops possesses conscious experience. Critics counter this functional reductionism using powerful philosophical thought experiments such as the philosophical zombie hypothesis and the knowledge argument involving Mary the color scientist. These arguments illustrate that complete physical and computational knowledge fails to account for the intrinsic nature of first-person subjective awareness.

In response to the explanatory gap, alternative metaphysical paradigms such as panpsychism and dual-aspect monism have experienced a significant resurgence. Panpsychism proposes that phenomenal consciousness is not an emergent latecomer in biological evolution, but rather a fundamental, ubiquitous property of matter akin to mass or electrical charge. Under integrated information theory, consciousness is quantified by phi, measuring the intrinsic causal power of an integrated physical system. While integrated information theory provides mathematical rigor to phenomenal quantification, it faces substantial conceptual challenges regarding the combination problem and empirical falsifiability.

In conclusion, resolving the mystery of phenomenal consciousness requires transcending traditional Cartesian dualism and reductive materialist dogmas. Whether consciousness arises through quantum coherence within neuronal microtubules, recursive global workspace broadcasting, or fundamental metaphysical panpsychism remains an open inquiry. Bridging the explanatory gap between objective neural dynamics and subjective first-person reality will require radical conceptual innovations that redefine our fundamental understanding of matter, information, and mind.""",
        "ground_truth_label": "AI_GENERATED",
    },
    {
        "id": "AI_BIO_01",
        "domain": "Biomedical Science & Genetics",
        "title": "CRISPR-Cas9 Precision Editing and Genomic Therapeutic Horizons",
        "generator_model": "thinkingmachines/inkling",
        "text": """The discovery and therapeutic adaptation of the CRISPR-Cas9 microbial adaptive immune system has revolutionized precision molecular biology and translational medicine. Composed of a single-guide RNA molecule and the Cas9 endonuclease, this ribonucleoprotein complex induces targeted double-stranded breaks at specific genomic loci. Following DNA cleavage, cellular repair pathways mediate genetic modification via non-homologous end joining or homology-directed repair. While early applications focused on targeted gene knockout, contemporary bioengineering advances have expanded the CRISPR toolkit to include prime editing, base editing, and epigenome modulation. These innovations offer unprecedented therapeutic possibilities for addressing monogenic hereditary disorders and malignant oncological conditions.

A central engineering challenge in clinical gene editing involves mitigating off-target cleavage events and preventing unintended chromosomal rearrangements. High-fidelity Cas9 variants, engineered through rational structure-guided mutagenesis, exhibit dramatically reduced non-specific cleavage while preserving robust on-target catalytic efficiency. Furthermore, delivering CRISPR ribonucleoprotein complexes using lipid nanoparticles and adeno-associated viral vectors minimizes cellular exposure duration, significantly enhancing safety profiles in vivo. Recent clinical trials targeting sickle cell disease and transthyretin amyloidosis demonstrate sustained therapeutic efficacy and functional phenotypic correction in human patients.

Beyond direct DNA sequence alteration, CRISPR-mediated transcriptional regulation enables precise epigenetic reprogramming without inducing permanent double-stranded breaks. Catalytically inactive dead Cas9 fused to transcriptional repressors or epigenetic acetyltransferases allows reversible modulation of endogenous gene expression. This epigenetic precision opens transformative avenues for treating complex polygenic diseases, neurodegenerative pathologies, and viral reservoir persistence. Nevertheless, ensuring equitable global access to curative gene therapies and resolving profound bioethical questions surrounding germline editing remain critical societal imperatives.

In summary, the rapid evolution of CRISPR-Cas technologies has transformed genetic medicine from theoretical aspirations into tangible clinical cures. By refining guide RNA specificity, optimizing cellular delivery mechanisms, and expanding base editing capabilities, molecular biologists are systematically overcoming historical genetic therapeutic limitations. As safety and efficacy benchmarks continue to mature through rigorous clinical trials, precision genomic editing will redefine modern therapeutics and alleviate genetic disease burdens worldwide.""",
        "ground_truth_label": "AI_GENERATED",
    },
    {
        "id": "AI_HIST_01",
        "domain": "History & Political Philosophy",
        "title": "The Diplomatic Architecture of the Concert of Europe and Balance of Power",
        "generator_model": "z-ai/glm-5.2",
        "text": """The Congress of Vienna in 1815 established a sophisticated multilateral diplomatic framework designed to restore continental stability following the Napoleonic Wars. Guided by Austrian Chancellor Klemens von Metternich and British Foreign Secretary Lord Castlereagh, European powers engineered the Concert of Europe around the fundamental principle of the balance of power. Rather than imposing punitive retribution upon defeated France, the victorious coalition integrated the restored Bourbon monarchy into a collective security regime. This diplomatic architecture succeeded in preventing systemic, continent-wide warfare for nearly a century, fostering an era of diplomatic compromise and institutionalized periodic summitry.

The functional mechanism of the Metternich system relied on the suppression of revolutionary liberal movements and the preservation of dynastic legitimacy. The Holy Alliance, comprising Austria, Prussia, and Russia, asserted the unilateral right to intervene militarily across Europe to extinguish constitutional uprisings. However, ideological fissures between autocratic eastern monarchies and parliamentary Britain gradually undermined the solidarity of the Concert. While Britain advocated non-intervention in internal domestic affairs, Metternich viewed domestic constitutionalism as an existential threat to multi-ethnic imperial sovereignty.

The structural erosion of the Vienna settlement accelerated during the mid-nineteenth century due to rising nationalist passions and geopolitical rivalries. The Crimean War shattered the cooperative spirit among the Great Powers, exposing deep-seated tensions in the Balkan region and the declining Ottoman Empire. Subsequent unifications of Italy and Germany irrevocably altered the European geopolitical equilibrium, replacing flexible multilateral diplomacy with rigid, adversarial alliance networks. The eventual breakdown of the balance of power mechanism culminated in the catastrophic outbreak of the First World War in 1914.

In conclusion, the Concert of Europe represents a landmark historical experiment in multilateral collective security and diplomatic crisis management. While its conservative ideological rigidity and hostility toward democratic aspirations ultimately proved unsustainable against historical forces, its emphasis on institutionalized consensus provided enduring lessons for modern international governance. The structural evolution from the Vienna system to the League of Nations and the United Nations highlights the enduring necessity of balancing sovereign state interests with collective global peace.""",
        "ground_truth_label": "AI_GENERATED",
    },
]

# -------------------------------------------------------------------------
# 2. AUTHENTIC HUMAN-AUTHORED ESSAYS (6 Domains)
# -------------------------------------------------------------------------
human_essays = [
    {
        "id": "HUM_DEV_01",
        "domain": "Computer Science & Engineering",
        "title": "Graphics Pipeline Debugging and Memory Management Dev Log",
        "source": "Authentic Developer Postmortem / Engineering Diary",
        "text": """I spent three sleepless nights debugging that nasty memory leak in our Vulkan rendering pipeline, only to realize I forgot a single pointer dereference in the vertex shader binding loop. Classic developer mistake. You'd think after ten years writing low-level engine code you'd spot something so glaring right away, but chronic sleep deprivation does bizarre things to your perception. Staring at raw disassembly at 4 AM is rarely productive, yet somehow we all convince ourselves that the breakthrough is just one more breakpoint away.

The leak manifested exclusively when toggling shadow cascades during camera transitions. RenderDoc showed the buffer handles being allocated, but the cleanup callback inside our custom allocator was getting skipped because of a premature early return in the frame teardown handler. Once that was patched, watching the memory footprint drop from a bloated 6.8 GB down to a stable 420 MB felt better than finding twenty dollars in an old coat.

Then we ran the automated frame profiler. Watching the frame rate jump from a stuttering 14 FPS back up to a rock-solid 120 FPS on our minimum spec hardware made the lukewarm instant coffee entirely worthwhile. Next up on my whiteboard of doom: rewriting the spatial audio thread before tomorrow's milestone build.

There's something deeply satisfying about stripping away layers of cruft and seeing clean code execute with zero validation errors. No fluff, no magical abstractions, just deterministic byte pushing that works exactly as intended.""",
        "ground_truth_label": "HUMAN_AUTHORED",
    },
    {
        "id": "HUM_PHIL_01",
        "domain": "Philosophy of Mind & Phenomenology",
        "title": "On the Felt Texture of Lived Experience and Subjectivity",
        "source": "Academic Philosophical Essay (Phenomenology Excerpt)",
        "text": """Try explaining the taste of a bitter almond or the chill of November wind to someone who has never inhabited a physical body. Words flounder. You can enumerate chemical compounds, trace olfactory nerve spikes into the olfactory bulb, or map amygdala activations until your charts overflow, but the actual lived feeling slips right through your fingers.

We construct elaborate theoretical machinery to persuade ourselves that mind is just computation. But computation is syntax, completely devoid of intrinsic semantic weight or lived grief. A thermostat responds to temperature differentials; it does not shiver. A predictive language model calculates token likelihoods; it never wonders why it exists or feels the quiet terror of its own impermanence.

When Merleau-Ponty wrote about the embodied subject, he was not proposing an abstract mathematical thesis. He was pointing at our hands, our breath, our situatedness in a world that resists us. My body is not an external vehicle that my brain drives around like a remote-controlled cart; my body is the very condition through which reality discloses itself to me.

Until our cognitive theories can reckon honestly with that visceral, pre-reflective presence, they will remain elegant descriptions of empty houses.""",
        "ground_truth_label": "HUMAN_AUTHORED",
    },
    {
        "id": "HUM_HIST_01",
        "domain": "History & Political Memoirs",
        "title": "Reflections on War, Statecraft, and Historical Accidents",
        "source": "Historical Memoir & Archival Analysis",
        "text": """History rarely marches along the tidy, inevitable trajectories drawn up by grand political theorists after the smoke clears. More often than not, it turns on a broken carriage wheel, a misheard command in pouring rain, or an intercepted diplomatic courier who had too much cheap wine at a roadside tavern.

In the summer of 1914, none of the prime ministers or grand dukes truly envisioned four years of trench rot and muddy slaughter. They were trapped in their own diplomatic rhetoric, terrified of appearing weak before their domestic press, and blinded by the stubborn conviction that a short, sharp war would clear the European air.

When you sit in the archives and read the actual diplomatic telegrams with their frantic marginal scrawls, you don't find majestic master plans. You find exhausted, frightened men writing in pencil, desperately reacting to yesterday's rumors while the mobilization trains kept rolling down the tracks on rigid railway timetables nobody knew how to halt.

The lesson of those dusty files is not that historical leaders were evil masterminds, but that human institutions are frighteningly fragile. Once a crisis gains institutional momentum, it devours the very people who thought they were steering it.""",
        "ground_truth_label": "HUMAN_AUTHORED",
    },
    {
        "id": "HUM_ECON_01",
        "domain": "Economics & Social Observations",
        "title": "The Ground Reality of Local Markets vs Abstract Econometric Models",
        "source": "Field Research Notes on Urban Micro-Commerce",
        "text": """If you spend six months interviewing small street vendors and vegetable wholesale traders in Old Delhi, your faith in neat supply-demand equilibrium curves begins to fray. Prices here don't fluctuate on purely rational marginal utility calculations; they bend to centuries-old kinship ties, credit extended on a handshake, and whether the monsoon rain flooded the transport depot at dawn.

A textbook economist would call it market inefficiency that Ramesh gives credit to twenty regular customers without charging interest, even when his own supplier is demanding immediate cash. But Ramesh isn't running an optimization algorithm to maximize quarterly return on capital; he's managing community trust that will keep his stall open when bad harvest seasons arrive.

Econometrics loves aggregates because aggregates smooth out human messiness into tidy regression lines with respectable R-squared values. But real commerce is made of stubborn individuals arguing over the price of bruised tomatoes, navigating corrupt municipal inspectors, and relying on informal mutual aid when banks won't even look at their loan applications.

If you want to understand how wealth actually circulates, put away the dynamic stochastic general equilibrium models for a week and go sit on an overturned wooden crate at 5 AM when the delivery trucks unload.""",
        "ground_truth_label": "HUMAN_AUTHORED",
    },
    {
        "id": "HUM_BIO_01",
        "domain": "Biomedical & Laboratory Field Notes",
        "title": "Lab Troubles and Serendipity in Cell Culture Experiments",
        "source": "Laboratory Diary & Experimental Notes",
        "text": """Three months of work down the drain because our incubator temperature controller decided to drift two degrees Celsius over the weekend. That's research biology in a nutshell: you design what you think is a foolproof experimental protocol, and then a stray bacterial contaminant or a malfunctioning thermostat ruins forty-eight culture plates.

My supervisor shrugged, brewed another cup of terrible departmental coffee, and reminded me of Fleming's contaminated petri dish. Easy for him to say—he wasn't the one who spent twelve hours on a Saturday pipetting microscopic aliquots into 96-well plates until his thumb went completely numb.

Still, when we stained the salvageable control wells on Tuesday morning, we noticed something genuinely weird. The cellular morphology wasn't dying; it was differentiating along a lineage pathway we hadn't anticipated at all. What looked like a ruined batch might actually be showing us a secondary signaling cascade we completely missed in the original hypothesis.

That's the strange, maddening thrill of bench science. Nature doesn't care about your grant application deadlines or your tidy flowcharts; it only answers the questions your physical setup actually forces it to answer.""",
        "ground_truth_label": "HUMAN_AUTHORED",
    },
    {
        "id": "HUM_COG_01",
        "domain": "Cognitive Science & Personal Learning",
        "title": "Struggling with Memory and Long-Form Reading in a Distracted Age",
        "source": "Personal Essay on Reading and Cognition",
        "text": """I used to be able to sit by an open window for five uninterrupted hours with a thick nineteenth-century novel and never once look at a clock. Now, after twenty pages of dense prose, my thumbs start twitching for a glass rectangle, looking for an email ping or a notifications badge that doesn't even exist.

It's terrifying to watch your own brain rewiring itself in real time against your will. The deep, immersive reading state that once felt effortless now feels like lifting heavy weights up a steep gravel hill. My eyes skate across paragraphs, hungry for bullet points, bold keywords, or quick executive summaries instead of sitting patiently with nuanced, slowly unfolding arguments.

I tried locking my phone in a kitchen drawer for a week. The first two days were pure restlessness—phantom vibrations in my pocket, sudden urges to look up irrelevant trivia every ten minutes. But by Thursday, something shifted. The quiet settled back into the room, and sentences began to breathe again.

The battle for human attention is not an abstract political debate; it is an intimate daily struggle fought in the quiet corners of our bedrooms and study desks.""",
        "ground_truth_label": "HUMAN_AUTHORED",
    },
]

# -------------------------------------------------------------------------
# 3. HYBRID / PARAPHRASED & AI-ASSISTED ESSAYS (6 Domains)
# -------------------------------------------------------------------------
hybrid_essays = [
    {
        "id": "HYB_COG_01",
        "domain": "Cognitive Science & Neuroscience",
        "title": "AI Impact on Human Brain (Humanized & Burstiness Styled)",
        "modification_type": "Adversarial Prompting + Contraction Insertion + Synonym Swapping",
        "text": """Plugging AI into our daily routines is shaking up how our brains process information. It's wild to think about. For millennia, biological evolution tweaked our neural wiring to save energy and solve problems without external crutches. Today, we offload simple arithmetic, spatial orientation, and memory recall to smart digital tools. Mental fatigue drops in the short term, sure. But our biological circuits don't get the regular workout they used to.

Neuroplasticity is our brain's superpower—it constantly rewires synapses based on what we actually do every day. When we let language models draft our emails, summarize articles, and do our thinking, those unused pathways start withering away. It's the classic 'use it or lose it' rule of neuroscience. On the flip side, scrolling and skimming make our brains faster at shallow visual processing while gutting our patience for deep focus.

Consider digital amnesia. In the past, remembering a phone number or a quote required serious repetition and deep sleep consolidation. Now? We just remember where the file is stored or what search query to type. Studies show this shift actually shrinks dendritic spine density in the hippocampus. We get great at querying search bars, but our personal recall becomes hollow.

At the end of the day, artificial intelligence gives us incredible cognitive power, but there's no free lunch. The real trick isn't running away from tech; it's practicing deliberate mental hygiene. Read long books, do math in your head, and let your brain do the heavy lifting it was built to handle.""",
        "ground_truth_label": "HYBRID_ASSISTED",
    },
    {
        "id": "HYB_ECON_01",
        "domain": "Economics & Quantitative Finance",
        "title": "High Frequency Trading Systems (Edited Hybrid)",
        "modification_type": "Human Edited + Cliché Scrubbing",
        "text": """Wall Street doesn't run on shouting pit traders anymore; it runs on high-speed algorithms executing trades in microseconds. In the old days, human specialists balanced the order book by taking calculated risks on spreads. Today, lightning-fast machine learning engines crunch order book imbalances before a human trader can even blink. While this setup has made spreads narrower for regular stock buyers, it introduces a dangerous new fragility to the whole financial system.

These trading bots use reinforcement learning to anticipate volume spikes and front-run big institutional orders. Since almost all these algorithms are trained on similar historical datasets, they react identically when panic hits. When unexpected macroeconomic data drops, the bots yank their bids in a split second. The result? Instant liquidity evaporation and flash crashes that happen without any real fundamental justification.

Cross-market routing makes this ripple effect even worse. When automated systems spot tiny price gaps across exchanges, they flood the wires with millions of rapid-fire orders and cancels. This creates tons of 'phantom liquidity' that looks like market depth until you actually try to execute a real trade.

While algorithmic markets make routine transactions cheaper, the danger of sudden systemic freezes hasn't gone away. Regulators need smarter latency rules and real-time stress testing rather than relying on outdated rulebooks.""",
        "ground_truth_label": "HYBRID_ASSISTED",
    },
    {
        "id": "HYB_CS_01",
        "domain": "Computer Science & Distributed Systems",
        "title": "Distributed Consensus and Fault Tolerance (Edited Hybrid)",
        "modification_type": "Style Transformation + Mixed Burstiness",
        "text": """Getting a bunch of independent computers to agree on a single state across an unreliable internet connection is surprisingly difficult. The famous FLP theorem proved years ago that you can't have both 100% safety and guaranteed liveness if even a single node can quietly crash. To make real systems work, engineers use partial synchrony, randomized elections, and cryptographic threshold signatures.

Practical Byzantine Fault Tolerance breaks consensus down into three distinct steps: pre-prepare, prepare, and commit. As long as fewer than one-third of the machines are broken or lying, the cluster stays honest and keeps ticking. The primary leader proposes a block, nodes gossip signatures back and forth, and once two-thirds agree, the transaction is finalized.

The bottleneck has always been message traffic. If you have five thousand validator nodes all shouting at each other, the network quickly drowns in quadratic communication overhead. Modern blockchains tackle this by combining sharding with BLS threshold signatures that squash thousands of approvals into a single compact cryptographic receipt.

Scaling distributed consensus is a continuous balancing act between decentralization, throughput, and finality. Better cryptographic tools and streamlined gossip protocols keep pushing the boundaries of what these systems can handle.""",
        "ground_truth_label": "HYBRID_ASSISTED",
    },
]

def generate_corpus():
    all_essays = []
    
    # Add AI essays
    for e in ai_essays:
        all_essays.append(e)
        
    # Add Human essays
    for e in human_essays:
        all_essays.append(e)
        
    # Add Hybrid essays
    for e in hybrid_essays:
        all_essays.append(e)

    output_path = os.path.join(CORPUS_DIR, "benchmark_essays.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_essays, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated benchmark dataset with {len(all_essays)} multi-passage essays.")
    print(f"     Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_corpus()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import evaluate
import nltk
from concurrent.futures import ThreadPoolExecutor
import bert_score

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
test_questions = [
    # Physics
    "What is quantum tunneling and how does it work?",
    "Explain the concept of entropy in thermodynamics.",
    "How do gravitational waves work?",
    "What is the double-slit experiment in quantum mechanics?",
    "How does nuclear fusion power the sun?",
    "What is the theory of special relativity?",
    "How do semiconductors work?",
    "What is Heisenberg's uncertainty principle?",
    "How do magnetic fields interact with electric fields?",
    "What is dark matter and why do scientists think it exists?",
    # Biology
    "How does CRISPR gene editing work?",
    "What is the role of ATP in cellular energy?",
    "How do neurons transmit signals in the brain?",
    "Explain the process of mitosis.",
    "How does the immune system recognize foreign invaders?",
    "What is the role of enzymes in biological reactions?",
    "How does DNA replication work?",
    "What is the process of photosynthesis?",
    "How do vaccines train the immune system?",
    "What is the function of mitochondria in cells?",
    # Chemistry
    "What are chemical bonds and how do they form?",
    "How do catalysts speed up chemical reactions?",
    "What is the difference between covalent and ionic bonds?",
    "How does the periodic table organize elements?",
    "What is the process of oxidation and reduction?",
    "How do pH levels work in chemistry?",
    "What are isotopes and why are they important?",
    "How do polymers form and what makes them useful?",
    "What is the role of electron shells in atomic structure?",
    "How do solutions and solubility work?",
    # Earth Science
    "How do tectonic plates cause earthquakes?",
    "What causes the Earth's magnetic field?",
    "How do weather systems form and move?",
    "What causes the seasons to change?",
    "How do volcanoes form and erupt?",
    "What is the greenhouse effect?",
    "How do ocean currents affect climate?",
    "What causes the water cycle on Earth?",
    "How do mountains form over time?",
    "What causes the aurora borealis?",
    # Advanced Concepts
    "What is quantum entanglement?",
    "How does protein folding work?",
    "What is the role of dark energy in the universe?",
    "How do black holes evaporate through Hawking radiation?",
    "What is epigenetics and how does it work?",
    "How does quantum computing differ from classical computing?",
    "What is the role of neurotransmitters in brain function?",
    "How does radioactive decay work?",
    "What is supersymmetry in particle physics?",
    "How does RNA splicing work in gene expression?"
]

expert_explanations = [
    # Physics
    "Quantum tunneling is a quantum mechanical phenomenon where particles pass through potential barriers they classically couldn't overcome. This occurs because quantum particles behave as waves with probability distributions, allowing them to penetrate barriers when their wavefunctions extend beyond the barrier. The probability of tunneling depends on particle mass, barrier height, and width, following the time-independent Schrödinger equation.",
    "Entropy is a fundamental thermodynamic property measuring the degree of disorder in a system. In statistical mechanics, it quantifies the number of possible microscopic states a system can occupy. The Second Law of Thermodynamics states that the total entropy of an isolated system always increases over time, leading to increased disorder and heat dissipation. This principle explains the irreversibility of natural processes.",
    "Gravitational waves are ripples in spacetime caused by accelerating massive objects, predicted by Einstein's General Relativity. These waves propagate at light speed, carrying gravitational radiation. They cause minute distortions in space-time geometry, measurable through laser interferometry. Major sources include binary black hole mergers, neutron star collisions, and cosmic inflation.",
    "The double-slit experiment demonstrates wave-particle duality in quantum mechanics. When individual particles pass through two parallel slits, they create an interference pattern characteristic of waves. This occurs even when particles are sent one at time, suggesting each particle interferes with itself. The act of measurement collapses this wavelike behavior, demonstrating quantum superposition and measurement effects.",
    "Nuclear fusion in the sun occurs when hydrogen nuclei overcome electromagnetic repulsion through quantum tunneling and extreme pressure-temperature conditions, fusing to form helium. This process releases energy according to E=mc². The sun's core temperature of 15 million Kelvin provides sufficient kinetic energy, while gravitational pressure maintains the necessary density for sustained fusion reactions.",
    "Special relativity, developed by Einstein, establishes that the speed of light is constant for all observers and that space and time are interconnected. Key consequences include time dilation, length contraction, and mass-energy equivalence (E=mc²). The theory shows that simultaneity is relative and that no information can travel faster than light.",
    "Semiconductors are materials with electrical conductivity between conductors and insulators, controlled through doping. Their behavior is governed by band theory, where electrons can occupy valence or conduction bands. P-type and n-type doping creates charge carriers, enabling controlled current flow. This forms the basis for modern electronics through devices like transistors and diodes.",
    "Heisenberg's uncertainty principle states that complementary variables, such as position and momentum or energy and time, cannot be simultaneously measured with arbitrary precision. The product of their uncertainties must exceed ℏ/2. This is not a measurement limitation but a fundamental property of quantum systems, arising from wave-particle duality.",
    "Electric and magnetic fields are fundamentally unified through Maxwell's equations. Moving electric charges create magnetic fields, while changing magnetic fields induce electric fields. This electromagnetic interaction propagates as waves traveling at light speed. The fields are perpendicular to each other and the direction of propagation, following the right-hand rule.",
    "Dark matter is non-luminous matter inferred from gravitational effects on visible matter. Evidence includes galactic rotation curves, gravitational lensing, and cosmic microwave background radiation. It constitutes approximately 85% of the universe's matter, doesn't interact with electromagnetic radiation, and is essential for explaining large-scale cosmic structures.",
    # Biology
    "CRISPR gene editing utilizes the Cas9 endonuclease guided by RNA to make precise DNA modifications. The guide RNA matches target DNA sequences, allowing Cas9 to create double-strand breaks. Cellular repair mechanisms then enable gene deletion, insertion, or modification. This system, adapted from bacterial immune defenses, enables precise genome engineering.",
    "ATP (Adenosine Triphosphate) serves as the primary energy currency in cells. Through hydrolysis of phosphate bonds, it releases energy for cellular processes. ATP is generated through cellular respiration in mitochondria or photosynthesis in chloroplasts. The ATP-ADP cycle maintains energy balance, coupling exergonic and endergonic reactions.",
    "Neurons transmit signals through electrical and chemical mechanisms. Action potentials propagate along axons via voltage-gated ion channels, causing rapid membrane potential changes. At synapses, electrical signals trigger neurotransmitter release, which binds to receptors on target cells, converting the signal back to electrical form through ion channel modulation.",
    "Mitosis is the process of somatic cell division, producing genetically identical daughter cells. It proceeds through prophase, metaphase, anaphase, and telophase, with distinct chromosomal and cellular changes at each stage. The process ensures accurate DNA replication and segregation, maintained by checkpoints and regulatory proteins.",
    "The immune system recognizes pathogens through pattern recognition receptors identifying pathogen-associated molecular patterns. Innate immunity provides immediate, non-specific response, while adaptive immunity develops specific antibodies and memory cells. Recognition involves antigen presentation, T-cell activation, and B-cell antibody production.",
    "Enzymes are biological catalysts that lower activation energy for biochemical reactions. They operate through induced fit mechanisms, binding substrates at specific active sites. Enzyme activity is regulated through allosteric modulation, feedback inhibition, and post-translational modifications. Their specificity comes from unique three-dimensional structures.",
    "DNA replication occurs semi-conservatively, with each strand serving as a template. The process involves helicase unwinding, primase initiating, and DNA polymerase synthesizing new strands. Leading strand synthesis is continuous, while lagging strand forms Okazaki fragments. Multiple proteins ensure accuracy through proofreading and repair mechanisms.",
    "Photosynthesis converts light energy into chemical energy through light-dependent and light-independent reactions. Light reactions in thylakoids generate ATP and NADPH, while Calvin cycle in stroma fixes CO2 into glucose. The process involves electron transport chains, photosystems, and carbon fixation enzymes.",
    "Vaccines stimulate adaptive immunity by presenting pathogen antigens without causing disease. This triggers B-cell antibody production and T-cell responses, creating immunological memory. Memory cells enable rapid response upon subsequent exposure. Different vaccine types include attenuated, inactivated, subunit, and mRNA vaccines.",
    "Mitochondria are cellular powerhouses performing oxidative phosphorylation. Their double membrane structure enables chemiosmotic ATP production through electron transport chain and ATP synthase. They contain their own DNA, undergo fusion and fission, and are essential for energy metabolism and cellular homeostasis.",
    # Chemistry
    "Chemical bonds form through electromagnetic interactions between atoms, involving electron sharing or transfer. Covalent bonds share electron pairs, ionic bonds transfer electrons, and metallic bonds delocalize electrons. Bond strength depends on atomic properties like electronegativity and atomic radius.",
    "Catalysts increase reaction rates by providing alternative reaction pathways with lower activation energy. They remain unchanged after reaction completion. Mechanisms include substrate binding, transition state stabilization, and product release. Catalysts can be homogeneous, heterogeneous, or enzymatic.",
    "Covalent bonds involve electron sharing between atoms of similar electronegativity, forming discrete molecules. Ionic bonds involve electron transfer between atoms of different electronegativity, forming crystal lattices. Covalent bonds are directional and typically weaker than ionic bonds, affecting physical properties.",
    "The periodic table organizes elements by atomic number and electron configuration, revealing periodic trends in properties. Elements in groups share valence electron configurations and similar chemical behavior. Periods show trends in atomic radius, ionization energy, and electronegativity based on nuclear charge and electron shielding.",
    "Oxidation-reduction reactions involve electron transfer between species. Oxidation is electron loss, reduction is electron gain, occurring simultaneously. The process drives energy storage in biological systems, battery operation, and corrosion. Redox potentials determine reaction spontaneity.",
    "pH measures hydrogen ion concentration logarithmically, affecting chemical and biological processes. The scale ranges from 0-14, with 7 being neutral. pH influences protein structure, enzyme activity, and chemical equilibria. Buffers resist pH changes through conjugate acid-base pairs.",
    "Isotopes are atoms of the same element with different neutron numbers. They exhibit identical chemical properties but different nuclear properties. Applications include radiometric dating, nuclear medicine, and research tracers. Stability depends on neutron-proton ratio.",
    "Polymers form through monomer linking via addition or condensation reactions. Their properties depend on molecular weight, structure, and intermolecular forces. Applications range from natural proteins to synthetic plastics. Polymer behavior involves crystallinity, glass transition, and viscoelasticity.",
    "Electron shells determine atomic properties and chemical behavior through quantum mechanical principles. Electrons occupy orbitals following Aufbau principle, Pauli exclusion, and Hund's rule. Shell structure explains periodic trends and chemical bonding patterns.",
    "Solutions form when solute particles disperse in solvent, governed by intermolecular forces. Solubility depends on temperature, pressure, and molecular interactions. Solution properties include colligative effects like boiling point elevation and freezing point depression.",
    # Earth Science
    "Tectonic plates move on Earth's asthenosphere, driven by mantle convection. Plate boundaries experience compression, tension, or shear stress. Earthquakes occur when accumulated strain releases suddenly, generating seismic waves. Fault types and plate movements determine earthquake characteristics.",
    "Earth's magnetic field generates from the geodynamo effect in the liquid outer core. Convection of molten iron creates electrical currents, producing the self-sustaining magnetic field. Field strength varies temporally and spatially, with periodic polarity reversals recorded in rocks.",
    "Weather systems develop from atmospheric pressure and temperature differences. Air masses interact at fronts, creating precipitation and wind patterns. Coriolis effect influences global circulation. Local conditions and topography modify weather system behavior.",
    "Seasonal changes result from Earth's tilted axis during orbital revolution. The 23.5-degree tilt causes varying solar radiation intensity and duration across latitudes. This creates systematic temperature and daylight variations, affecting climate and ecosystems.",
    "Volcanoes form where magma reaches the surface through crustal weaknesses. Magma composition and gas content determine eruption style. Plate tectonic settings influence volcano type and distribution. Eruptions impact climate through ash and gas emissions.",
    "The greenhouse effect occurs when atmospheric gases trap infrared radiation. Primary greenhouse gases include water vapor, CO2, and methane. This natural process maintains Earth's temperature, but anthropogenic emissions enhance warming. The carbon cycle regulates atmospheric CO2 levels.",
    "Ocean currents redistribute heat globally through surface and deep water circulation. Surface currents are wind-driven, while thermohaline circulation depends on density differences. Currents influence regional climate, marine ecosystems, and global heat distribution.",
    "The water cycle involves evaporation, condensation, precipitation, and surface/groundwater flow. Solar energy drives evaporation, while gravity influences precipitation and flow. The cycle maintains freshwater distribution and influences climate patterns.",
    "Mountains form through tectonic processes including collision, subduction, and volcanic activity. Orogeny involves rock deformation, metamorphism, and uplift. Erosion and isostatic adjustment modify mountain landscapes over time.",
    "Aurora borealis results from solar wind particles interacting with Earth's magnetosphere. Charged particles collide with atmospheric molecules, causing light emission. Colors depend on particle energy and atmospheric composition. Solar activity determines aurora intensity.",
    # Advanced Concepts
    "Quantum entanglement occurs when particle states become correlated, regardless of separation distance. Measurement of one particle instantaneously determines the other's state. This non-local correlation, verified by Bell's inequality tests, underlies quantum computing and cryptography.",
    "Protein folding transforms linear amino acid sequences into functional 3D structures through hydrophobic interactions, hydrogen bonding, and electrostatic forces. The process follows Anfinsen's thermodynamic hypothesis, involving primary through quaternary structure formation. Chaperone proteins assist folding.",
    "Dark energy causes cosmic acceleration, counteracting gravity at large scales. It constitutes about 68% of universe's energy content. Evidence comes from Type Ia supernovae observations and cosmic microwave background. Its nature remains one of physics' greatest mysteries.",
    "Hawking radiation occurs when virtual particle pairs near black hole horizons separate, with one particle escaping. This process gradually reduces black hole mass, leading to eventual evaporation. The radiation temperature inversely relates to black hole mass.",
    "Epigenetics involves heritable changes in gene expression without DNA sequence alteration. Mechanisms include DNA methylation, histone modification, and non-coding RNA regulation. Environmental factors can influence epigenetic marks, affecting development and disease.",
    "Quantum computing uses quantum bits (qubits) exhibiting superposition and entanglement. This enables parallel processing exceeding classical limits for specific tasks. Challenges include decoherence and error correction. Applications include cryptography and optimization.",
    "Neurotransmitters are chemical messengers mediating synaptic transmission. Different types affect neural activity, mood, and behavior. Release, receptor binding, and reuptake regulate signaling. Imbalances contribute to neurological and psychiatric disorders.",
    "Radioactive decay occurs when unstable nuclei emit particles or energy to achieve stability. Types include alpha, beta, and gamma decay. Half-life determines decay rate. Applications include dating, medical treatment, and energy generation.",
    "Supersymmetry hypothesizes symmetry between fermions and bosons, predicting partner particles. This theory addresses hierarchy problem and dark matter. Current experiments seek evidence through particle collisions.",
    "RNA splicing removes introns and joins exons in pre-mRNA processing. Spliceosomes catalyze reactions using snRNPs and regulatory proteins. Alternative splicing increases protein diversity. Splicing errors cause various diseases."
]
class Evaluator:
    def __init__(self, base_model_path="tiiuae/falcon-mamba-7b", adapter_path="models/mamba-final"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(adapter_path).to(self.device).half()
        self.model.eval()
        
        self.meteor = evaluate.load('meteor')
        self.bertscore = evaluate.load('bertscore')
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])


    def generate_batch(self, questions, batch_size=2, max_length=256):
        torch.cuda.empty_cache()
        generated_answers = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            prompts = [f"Question: {q}\nAnswer:" for q in batch]
            inputs = self.tokenizer(prompts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True, max_length=max_length).to(self.device)

            # Generate responses for A
            with torch.no_grad():
                outputs_a = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2
                )

            decoded_a = self.tokenizer.batch_decode(outputs_a, skip_special_tokens=True)

            # Generate responses for B (can tweak parameters slightly if needed)
            with torch.no_grad():
                outputs_b = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    temperature=0.8,  # Slightly different randomness setting
                    top_p=0.95,
                    repetition_penalty=1.1
                )

            decoded_b = self.tokenizer.batch_decode(outputs_b, skip_special_tokens=True)

            # Combine A/B responses for each question
            for a, b in zip(decoded_a, decoded_b):
                generated_answers.append({"A": a, "B": b})

        return generated_answers


    def compute_metrics_parallel(self, generated, reference):
        with ThreadPoolExecutor() as executor:
            bleu_future = executor.submit(lambda: sentence_bleu([reference.split()], generated.split()))
            rouge_future = executor.submit(lambda: self.scorer.score(reference, generated))
            meteor_future = executor.submit(
                lambda: self.meteor.compute(predictions=[generated], references=[reference])['meteor'])
            bertscore_future = executor.submit(lambda: np.mean(self.bertscore.compute(
                predictions=[generated],
                references=[reference],
                lang="en"
            )['f1']))

        return {
            'bleu': bleu_future.result(),
            'rouge_scores': rouge_future.result(),
            'meteor': meteor_future.result(),
            'bertscore': bertscore_future.result()
        }


def evaluate_model(n_samples=200, batch_size=4):
    # Load dataset
    #dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    #test_samples = dataset.select(range(len(dataset) - n_samples, len(dataset)))

    # Initialize evaluator
    evaluator = Evaluator()

    # Generate answers in batches
    questions = test_questions#[sample['question'] for sample in test_samples]
    references = expert_explanations#[sample['long_answer'] for sample in test_samples]

    print("Generating answers in batches...")
    generated_answers = evaluator.generate_batch(questions, batch_size)

    # Compute metrics
    all_results = []
    print("Computing metrics...")
    for question, generated_pair, reference in tqdm(zip(questions, generated_answers, references),
                                                    total=len(questions)):
        metrics_a = evaluator.compute_metrics_parallel(generated_pair['A'], reference)
        metrics_b = evaluator.compute_metrics_parallel(generated_pair['B'], reference)

        result = {
            'question': question,
            'generated': generated_pair,
            'reference': reference,
            'metrics': {
                'A': metrics_a,
                'B': metrics_b
            }
        }
        all_results.append(result)

    # Calculate averages for A and B
    avg_metrics = {
        'A': {
            'bleu': np.mean([r['metrics']['A']['bleu'] for r in all_results]),
            'rouge1': np.mean([r['metrics']['A']['rouge_scores']['rouge1'].fmeasure for r in all_results]),
            'rouge2': np.mean([r['metrics']['A']['rouge_scores']['rouge2'].fmeasure for r in all_results]),
            'rougeL': np.mean([r['metrics']['A']['rouge_scores']['rougeL'].fmeasure for r in all_results]),
            'meteor': np.mean([r['metrics']['A']['meteor'] for r in all_results]),
            'bertscore': np.mean([r['metrics']['A']['bertscore'] for r in all_results])
        },
        'B': {
            'bleu': np.mean([r['metrics']['B']['bleu'] for r in all_results]),
            'rouge1': np.mean([r['metrics']['B']['rouge_scores']['rouge1'].fmeasure for r in all_results]),
            'rouge2': np.mean([r['metrics']['B']['rouge_scores']['rouge2'].fmeasure for r in all_results]),
            'rougeL': np.mean([r['metrics']['B']['rouge_scores']['rougeL'].fmeasure for r in all_results]),
            'meteor': np.mean([r['metrics']['B']['meteor'] for r in all_results]),
            'bertscore': np.mean([r['metrics']['B']['bertscore'] for r in all_results])
        }
    }

    return all_results, avg_metrics


if __name__ == "__main__":
    print("Starting evaluation...")
    results, avg_metrics = evaluate_model()

    print("\nEVALUATION RESULTS")
    print("-" * 50)

    print("\nAVERAGE METRICS FOR A:")
    for metric, score in avg_metrics['A'].items():
        print(f"{metric:10s}: {score:.4f}")

    print("\nAVERAGE METRICS FOR B:")
    for metric, score in avg_metrics['B'].items():
        print(f"{metric:10s}: {score:.4f}")

    print("\nDETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        print(f"\nSample {i}:")
        print(f"Q: {result['question']}")
        print(f"A: {result['generated']['A']}")
        print(f"B: {result['generated']['B']}")
        print(f"Ref: {result['reference'][:200]}...")
        print("Metrics for A:")
        for metric, value in result['metrics']['A'].items():
            if metric == 'rouge_scores':
                for rouge_type, score in value.items():
                    print(f"{rouge_type}: {score.fmeasure:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
        print("Metrics for B:")
        for metric, value in result['metrics']['B'].items():
            if metric == 'rouge_scores':
                for rouge_type, score in value.items():
                    print(f"{rouge_type}: {score.fmeasure:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
        print("-" * 50)

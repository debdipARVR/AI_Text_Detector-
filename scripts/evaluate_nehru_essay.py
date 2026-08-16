"""Evaluate Jawaharlal Nehru multi-passage essay with Pass 2 & Pass 3 Cloze Congruence."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient

nehru_essay = """
# Jawaharlal Nehru: The Architect of Modern India

## Essay

Jawaharlal Nehru was one of the most influential leaders in the history of modern India. He was a freedom fighter, statesman, author, and the first Prime Minister of independent India. His vision, leadership, and commitment to democracy helped shape the newly independent nation. He devoted his entire life to serving the country and worked tirelessly for the welfare of its people. Known for his progressive ideas and love for children, Nehru occupies a special place in the hearts of Indians. He is often referred to as the "Architect of Modern India" because he laid the foundations of India's scientific, industrial, educational, and democratic development.

Jawaharlal Nehru was born on 14 November 1889 in Allahabad (now Prayagraj), Uttar Pradesh, into a wealthy and respected Kashmiri Brahmin family. His father, Motilal Nehru, was one of India's most successful lawyers and a prominent leader of the Indian National Congress. His mother, Swarup Rani Nehru, was a kind and religious woman who greatly influenced his character. Since his family was well educated and financially prosperous, Nehru received an excellent education from an early age. Private tutors taught him at home, where he developed a deep interest in literature, history, science, and nature.

At the age of fifteen, Nehru went to England for higher studies. He studied at Harrow School, one of the most prestigious schools in Britain. Later, he joined Trinity College, Cambridge, where he studied Natural Sciences. After completing his graduation, he enrolled at the Inner Temple, London, to study law and qualified as a barrister. During his stay in England, he was exposed to liberal ideas, democratic values, and the political movements taking place across Europe. These experiences greatly influenced his thinking and later shaped his vision for India.

After returning to India in 1912, Nehru began practicing law in Allahabad. However, he soon realized that his true calling was not the courtroom but the service of the nation. The political atmosphere in India was changing rapidly under British rule, and Nehru felt compelled to join the struggle for independence. His meeting with Mahatma Gandhi in 1916 marked a turning point in his life. Inspired by Gandhi's ideals of truth, non-violence, and civil disobedience, Nehru dedicated himself wholeheartedly to the Indian freedom movement.

Jawaharlal Nehru actively participated in several important movements launched by the Indian National Congress. He took part in the Non-Cooperation Movement (1920–22), the Civil Disobedience Movement (1930–34), and the Quit India Movement (1942). Throughout these movements, he was arrested and imprisoned by the British government many times. In total, he spent nearly nine years in prison. Despite imprisonment and hardships, Nehru never gave up his fight for freedom. Instead, he used his time in jail to read extensively and write books that reflected his deep understanding of history, politics, and civilization.

One of the most significant moments in Nehru's political career came in 1929, when he became the President of the Indian National Congress at the Lahore Session. Under his leadership, the Congress declared "Purna Swaraj" (Complete Independence) as its ultimate goal. On 26 January 1930, the Indian National Congress observed Independence Day, demanding complete freedom from British rule. This event inspired millions of Indians to intensify the struggle against colonial rule.

India finally achieved independence on 15 August 1947, after nearly two centuries of British rule. Jawaharlal Nehru became the first Prime Minister of independent India. His historic speech, "Tryst with Destiny," delivered at midnight on the eve of independence, remains one of the greatest speeches in history. In this speech, he spoke about India's responsibility to build a just, peaceful, and prosperous nation. His words inspired hope and confidence among millions of Indians who looked forward to a brighter future.

As Prime Minister, Nehru faced enormous challenges. India had just been partitioned into India and Pakistan, leading to widespread violence, migration, and communal tension. The economy was weak, industries were underdeveloped, literacy was low, and poverty was widespread. Nehru understood that political independence alone was not enough. He believed that India needed economic development, scientific progress, and social reforms to become a strong and self-reliant nation.

One of Nehru's greatest achievements was strengthening India's democratic system. He firmly believed that democracy, secularism, and equality should form the foundation of the nation. Under his leadership, India successfully conducted its first general elections in 1951–52, one of the largest democratic exercises in the world. He respected freedom of speech, encouraged parliamentary debates, and ensured that democratic institutions remained strong. His commitment to democracy helped India emerge as one of the world's largest and most stable democracies.

Nehru strongly believed that education was the key to national development. He worked tirelessly to establish world-class educational institutions. During his tenure, institutions such as the Indian Institutes of Technology (IITs), the All India Institute of Medical Sciences (AIIMS), the Indian Institutes of Management (conceptual groundwork), the University Grants Commission (UGC), and several national laboratories were established or strengthened. These institutions have produced scientists, engineers, doctors, researchers, and leaders who have contributed significantly to India's growth and global reputation.

Science and technology occupied a central place in Nehru's vision for India. He encouraged scientific research and promoted what he called the "scientific temper." He believed that scientific thinking, rationality, and innovation were essential for solving the country's problems. Under his leadership, organizations such as the Council of Scientific and Industrial Research (CSIR), the Atomic Energy Commission, and major research laboratories expanded their activities. India also took its first important steps in atomic energy and space research during his period of leadership.

Nehru also emphasized industrial development. He believed that heavy industries would provide the foundation for India's economic independence. His government established several steel plants, machine-building factories, fertilizer plants, oil refineries, and public-sector enterprises. He also introduced the Five-Year Plans to guide planned economic development. These plans focused on agriculture, irrigation, industries, transportation, and infrastructure. Large multipurpose river valley projects such as the Bhakra Nangal Dam, Hirakud Dam, and Damodar Valley Project were constructed during his tenure. Nehru famously referred to these dams as the "Temples of Modern India" because they symbolized progress and development.

In foreign policy, Jawaharlal Nehru adopted the principles of peaceful coexistence, anti-colonialism, and non-alignment. Along with leaders such as Josip Broz Tito of Yugoslavia and Gamal Abdel Nasser of Egypt, he played a leading role in establishing the Non-Aligned Movement (NAM). During the Cold War, Nehru believed that India should remain independent and not become part of either the American or Soviet military blocs. This policy helped India maintain its sovereignty and develop friendly relations with many countries across the world.

Despite many achievements, Nehru's tenure also faced serious challenges. The 1962 Sino-Indian War with China was a major setback. India's military suffered defeat, and Nehru's foreign policy came under criticism. The conflict deeply affected him personally and politically. Nevertheless, historians acknowledge that governing a newly independent and diverse country of hundreds of millions of people was an extraordinarily difficult task, and many of his long-term institutions continued to benefit India.

Jawaharlal Nehru had immense affection for children. He believed that children represented the future of the nation and deserved love, education, and opportunities to develop their talents. Because of his warmth and kindness, children affectionately called him "Chacha Nehru." To honor his love for children, 14 November, his birthday, is celebrated every year as Children's Day across India. Schools organize cultural programs, competitions, and educational activities to commemorate his legacy.

Apart from being a political leader, Nehru was an accomplished writer and thinker. His books continue to be widely read across the world. Among his most famous works are "The Discovery of India," "Glimpses of World History," "An Autobiography," and "Letters from a Father to His Daughter." These books reflect his deep knowledge of history, philosophy, culture, and politics. They also reveal his love for India and his hope for a modern, progressive, and united nation.

Jawaharlal Nehru passed away on 27 May 1964 after serving as Prime Minister for nearly seventeen years. His death marked the end of an important chapter in Indian history. Millions of people mourned the loss of a leader who had devoted his life to the nation. Even today, his ideas on democracy, secularism, scientific thinking, education, and international peace continue to influence India's development.

Jawaharlal Nehru's legacy remains both significant and widely discussed. Supporters praise him for establishing democratic institutions, promoting education, encouraging scientific research, and laying the foundation for industrial growth. Critics debate some of his economic and foreign policy decisions, particularly regarding China and aspects of the planned economy. Such debates are natural for a leader of his stature and reflect the lasting importance of his role in Indian history.

## Conclusion

In conclusion, Jawaharlal Nehru was not merely India's first Prime Minister but also a visionary nation-builder whose influence extended far beyond politics. He dreamed of an India that was democratic, secular, educated, scientifically advanced, and economically self-reliant. Through his leadership, institutions, and policies, he laid the groundwork for many of the achievements that India enjoys today. His lifelong dedication to freedom, social justice, education, and national development makes him one of the greatest leaders in Indian history. His life continues to inspire generations of Indians to work for the progress, unity, and prosperity of the nation.
"""

def main():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)
    res = detector.analyze(nehru_essay)

    print("=" * 80)
    print(" JAWAHARLAL NEHRU ESSAY: PASS 2 & PASS 3 DEEPEVAL EVALUATION")
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

    print(f"\n[Pass 3: Middle 3-Sentence Passage Removal] ({res['pass_3']['sentences_masked_count']} blanks across passages):")
    print(f"  Congruence Score:         {res['pass_3']['congruence_score']}%")
    print(f"  Meaning Similarity (40%): {res['pass_3']['meaning_similarity']}%")
    print(f"  Semantic Cosine (40%):    {res['pass_3']['semantic_cosine']}%")
    print(f"  Semantic Alignment (10%): {res['pass_3']['semantic_similarity']}%")
    print(f"  Lexical Overlap (10%):    {res['pass_3']['lexical_similarity']}%")
    print("  Sample Pass 3 Middle Sentence Pairs:")
    for s in res['pass_3']['spans'][:6]:
        print(f"    - Key {s['placeholder']} (Passage {s.get('paragraph_idx', 0)+1}):")
        print(f"        Original:  \"{s['original_sentence'][:65]}...\"")
        print(f"        AI Infill: \"{s['predicted_sentence'][:65]}...\"")
        print(f"        Scores:    Meaning={s['meaning_similarity']}% | Cosine={s['semantic_cosine']}% | Congruence={s['congruence']}%")
    print("=" * 80)

if __name__ == "__main__":
    main()

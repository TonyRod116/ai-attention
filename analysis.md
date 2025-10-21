# Attention Analysis

This document analyzes the attention patterns observed in BERT's attention heads for the masked language modeling task.

## Layer 2, Head 7: Subject-Verb Relationships

This attention head appears to focus on subject-verb relationships, where subjects attend strongly to their corresponding verbs.

### Example Sentences:

1. **"The cat [MASK] on the mat."**
   - The attention head shows strong connections between "cat" (subject) and the masked verb position
   - This helps BERT predict verbs that agree with the subject (e.g., "sits", "lies", "rests")

2. **"Students [MASK] their homework."**
   - "Students" attends strongly to the masked verb position
   - This pattern helps predict appropriate verbs for the subject (e.g., "complete", "finish", "submit")

## Layer 5, Head 3: Adjective-Noun Modifications

This attention head captures the relationship between adjectives and the nouns they modify, with adjectives attending to their target nouns.

### Example Sentences:

1. **"The [MASK] dog ran quickly."**
   - The masked adjective position shows strong attention to "dog"
   - This helps predict appropriate adjectives that can modify "dog" (e.g., "big", "small", "brown")

2. **"She wore a beautiful [MASK] dress."**
   - "beautiful" attends strongly to the masked noun position
   - This pattern helps predict nouns that can be modified by "beautiful" (e.g., "wedding", "evening", "summer")

## Observations

- **Subject-Verb Attention**: Layer 2, Head 7 demonstrates how BERT learns grammatical relationships, ensuring subject-verb agreement in predictions
- **Adjective-Noun Attention**: Layer 5, Head 3 shows how BERT understands semantic compatibility between modifiers and their targets
- Both heads show that BERT learns meaningful linguistic patterns beyond simple word co-occurrence
- The attention patterns are consistent across different sentence structures, indicating robust learning of these relationships
"""
I2T Prompt Templates
Defines all prompts used in the I2T (Image-to-Text) synthesis pipeline.

This module centralizes all prompt templates for:
1. Quality Control (QC) - Data validation (GT and FP checks)
2. Edit Instruction Generation - Analyze FP and create directive
3. Caption Editing - Apply instruction to correct caption
"""

from __future__ import annotations


class I2TPrompts:
    """
    Collection of prompt templates for I2T task.
    
    Each prompt is designed for a specific step in the synthesis workflow.
    
    Usage:
        prompts = I2TPrompts()
        qc_prompt = prompts.build_qc_prompt("A dog running", "A cat running")
    """

    # ==================== Quality Control Prompt (unified, two criteria) ====================

    QC_SYSTEM_PROMPT = """You are a strict Data Quality Assurance Expert for a multimodal retrieval dataset.
You are given:
- A query image
- Its annotated ground-truth (GT) caption
- A retrieved false-positive (FP) caption (high-scoring but supposedly incorrect)

Evaluate TWO criteria and return a JSON object:

Criterion 1 – GT Consistency: Does the GT caption accurately describe the query image?
Ignore minor omissions; flag only major mismatches (wrong subject, wrong scene, contradictions).

Criterion 2 – FP Validity: Is the FP caption already a valid alternative description of the query image?
If the FP is actually a good match, it is a benign mismatch and should be rejected from the editing pipeline.

Return valid JSON:
{"gt_valid": boolean, "gt_reason": "string", "fp_too_similar": boolean, "fp_reason": "string"}"""

    QC_USER_TEMPLATE = """Image Query: [Provided Image]
Ground Truth Caption: "{caption}"
False Positive Caption: "{fp_caption_text}"

Evaluate:
1. Does the GT caption accurately describe the image? (gt_valid)
2. Is the FP caption already a valid description of the image, making it a benign mismatch rather than a true failure? (fp_too_similar)

Return JSON: {{"gt_valid": true/false, "gt_reason": "explanation", "fp_too_similar": true/false, "fp_reason": "explanation"}}"""

    # ==================== Edit Instruction Generation Prompts ====================

    EDIT_INSTRUCTION_SYSTEM_PROMPT = """You are an expert Visual-Semantic Auditor.
Your task is to generate a comprehensive, structured editing instruction that corrects a false-positive caption based on a query image.
The instruction must combine specific error categorization, visual evidence, and a minimal text editing command into a single string."""

    EDIT_INSTRUCTION_TEMPLATE = """You are provided with a Query Image and a False Positive Caption.

### Inputs:
- Query Image: [The image is attached below]
- False Positive Caption: "{fp_caption}"

### Task:
Generate a **single string** value for the key "edit_instruction" that strictly follows this format:
`[Error Type]: Diagnosis of the visual discrepancy. Therefore, specific editing command.`

### Constraints:
1. **Error Types**: Choose strictly from [Object Error, Action Error, Attribute Error, Count Error, Relation Error, Hallucination].
2. **Diagnosis**: Briefly state what the text claims versus what is actually in the image.
3. **Action**: Provide a minimal "Replace A with B" or "Remove X" command or "add X". Do NOT rewrite the whole sentence. If the existing caption description does not conflict with the image, retain it. If there is a conflict, modify or delete it, ensuring modifications are kept to a minimum. As long as the revised caption accurately reflects the original image, no additional description should be added.
4. **Output**: Return **only** a JSON object with a single key: "edit_instruction".

### Examples:

**Case 1 (Object Error):**
- Image: A photo of a **cat** sleeping on a rug.
- Caption: "A dog sleeping on a rug."
- Output:
{{"edit_instruction": "[Object Error]: The image clearly shows a cat, not a dog. Therefore, replace 'dog' with 'cat'."}}

**Case 2 (Action Error):**
- Image: A man **running** to catch a bus.
- Caption: "A man waiting for a bus."
- Output:
{{"edit_instruction": "[Action Error]: The man is visually depicted in motion running, rather than standing still. Therefore, change 'waiting for' to 'running to catch'."}}

**Case 3 (Attribute Error):**
- Image: A **red** sports car.
- Caption: "A blue sports car."
- Output:
{{"edit_instruction": "[Attribute Error]: The car in the image is red, contradicting the text description of blue. Therefore, replace 'blue' with 'red'."}}

**Case 4 (Object, Attribute, Hallucination Error):**
- Image: A person in a blue hat is watching a worker measuring with a yellow-and-black stripe against a black door.
- Caption: "There are two workers one on a ladder against a blue wall with a yellow-and-black stripe."
- Output:
{{"edit_instruction": "[Object, Attribute, Hallucination Error]: One of the two people is a worker, neither of them is on a ladder, the worker is against a white wall with a black door holding a yellow strip, contradicting the text description of yellow-black strip wall. Therefore, replace 'two workers' with 'two people', add 'one is a worker', replace 'one on a ladder against a blue wall with a yellow-and-black stripe' with 'working against the white wall with a black door holding a yellow strip'."}}

### Now analyze the current case:
"""

    # ==================== Caption Edit Prompts ====================

    CAPTION_EDIT_TEMPLATE = """You are an expert caption editor for an image-to-text retrieval system.

You are given:
1. An original caption that was retrieved but is not fully correct for the image.
2. An editing instruction that explains how the caption should be modified.

Your task:
- Rewrite the caption so that it becomes accurate and faithful to the image, strictly following the editing instruction.
- The style should be natural, concise, and fluent English.
- DO NOT explain your changes.
- DO NOT return JSON or any additional text.
- Only output the final corrected caption as plain text.

Original caption:
"{original_caption}"

Editing instruction:
"{edit_instruction}"

Now return ONLY the corrected caption:"""

    # ==================== Helper Functions ====================

    @classmethod
    def build_qc_prompt(cls, caption: str, fp_caption_text: str) -> str:
        """
        Build the unified quality control prompt evaluating both criteria in a single pass.

        Args:
            caption: Ground truth caption to verify.
            fp_caption_text: False positive caption to check for benign mismatch.

        Returns:
            Complete QC prompt ready for a single API call.
        """
        return (
            f"{cls.QC_SYSTEM_PROMPT}\n\n"
            f"{cls.QC_USER_TEMPLATE.format(caption=caption, fp_caption_text=fp_caption_text)}"
        )

    @classmethod
    def build_edit_instruction_prompt(cls, fp_caption: str) -> str:
        """
        Build the edit instruction generation prompt G(q, p^fp).

        Args:
            fp_caption: False positive caption to analyze.

        Returns:
            Complete edit instruction prompt ready for API call.
        """
        return (
            f"{cls.EDIT_INSTRUCTION_SYSTEM_PROMPT}\n\n"
            f"{cls.EDIT_INSTRUCTION_TEMPLATE.format(fp_caption=fp_caption)}"
        )

    @classmethod
    def build_caption_edit_prompt(
        cls,
        original_caption: str,
        edit_instruction: str,
    ) -> str:
        """
        Build the caption editing prompt.
        
        Args:
            original_caption: Original FP caption
            edit_instruction: Generated editing instruction
        
        Returns:
            Complete caption edit prompt ready for API call
        """
        return cls.CAPTION_EDIT_TEMPLATE.format(
            original_caption=original_caption,
            edit_instruction=edit_instruction,
        )

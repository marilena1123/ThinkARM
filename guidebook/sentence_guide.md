# Sentence-level Annotation Guidebook

In this project, we aim to analyze the reasoning process of current large language models (LLMs) with advanced reasoning capabilities, i.e., Large Reasoning Models (LRMs), based on a modified version of Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. The original Schoenfeld theory was built on hundreds of hours of recorded tapes of students tackling non-routine math problems while being asked to think aloud. Widely regarded as a gold-standard framework in mathematics education research, this theory offers a rigorously validated, fine-grained lens for dissecting both expert and novice problem-solving strategies. After thorough investigation, we find that the thinking process of LRMs can be well-aligned with the episodes in the theory, as they also follow similar problem-solving processes. Thus, in this project, we aim to annotate the model (solver) responses with these episode categories. To better apply the theory to the analysis of model responses, we utilize sentence-level annotation, which is used to capture the fine-grained behavior of each sentence, including **eight** categories: Read, Analyze, Plan, Implement, Explore, Verify, Monitor, and Answer. The original Schoenfeld theory only has six categories: Read, Analyze, Plan, Implement, Explore, and Verify. These categories describe the thinking behaviors of how humans solve a problem. Later, an additional "Monitor" category was included in the system to capture behaviors that do not contain specific content but are still important, such as "Let me think." Moreover, when trying to apply the theory to analyzing LLM behaviors, we introduce another category, "Answer," to represent the sentence that delivers the answer. Thus, in total, there are eight categories. For each sentence, the annotation depends on both the current sentence itself and its context.

## **Label Definitions and Guidelines**

**1. Read**

* **Definition:** This is usually the initial phase, which focuses on extracting or restating the given information, conditions, and the goal of the problem as presented. It involves understanding the question without any inference of strategy or reasoning.
* **Guidelines:**
  * Sentences in this category should directly present the content of the original problem statement.
  * Look for phrases that recall or repeat elements of the question.
  * This label is mostly presented for the model's initial processing of the problem.
* **Potential Keywords/Indicators:** "The question asks...", "The problem requires...", "We are given...", "The goal is to...", "The choices are...", direct quotes from the problem.
* **Distinguishing Features:**
  * This stage is purely about understanding the input, not about processing it or deciding how to solve it. It should not contain content like trying to understand or analyze the question.
  * Avoid labeling sentences as Read if they include any form of analysis or evaluation of the problem. The Read stage usually appears at the beginning of the reasoning. However, it can also appear in the middle of the reasoning, in order to ensure that the question was understood correctly.
* **Example:** "The question asks us to find the value of x in the equation 2x + 5 = 10."

**2. Analyze**

* **Definition:** This stage involves constructing or recalling relevant theories, introducing necessary symbols, and deducing relationships based on the problem statement and existing knowledge. The core activity is explanation or logical inference that sets the stage for the solution but does not involve concrete calculations yet.
* **Guidelines:**
  * Sentences should explain the underlying mathematical concepts or principles relevant to the problem.
  * This category includes analysis of either the problem itself or intermediate results.
  * This label applies to logical deductions and inferences made with certainty.
* **Potential Keywords/Indicators:** "According to...", "We can define...", "This implies that...", "Therefore...", "Based on this...", "We can infer that...", "Let's note that...", "Let me observe that...", "Let's recall that..."
* **Distinguishing Features:**
  * The Analyze episode involves certain inferences and explanations, unlike Explore, which shows uncertainty.
  * The actual execution of calculations is mainly in the Implement stage.
  * Analyze does not involve any concrete calculation, which is unlike Implement.
  * The behavior of setting some basic notations should be in the Implement stage, e.g., "Let me set a=1, then...".
* **Important Note:** Be careful not to include sentences that involve substituting values or performing calculations, as those belong to the Implement stage.
* **Example:** "According to the Pythagorean theorem, in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides." or "If I can get the equation in slope-intercept form (y = mx + b), then I can plug in y = 4 and solve for x, which should be d."

**3. Plan**

* **Definition:** This stage involves announcing the next step or outlining the entire solution strategy. It represents a commitment to a particular course of action before the actual execution begins.
* **Guidelines:**
  * Sentences should clearly state the intended next step or the overall plan.
  * Look for explicit declarations of intent, often using the first person or imperative voice.
  * This stage signifies that a decision has been made on how to proceed, and the next step should be related to math problem solving, rather than generally saying "let's think about it."
* **Potential Keywords/Indicators:** "Next, we will...", "The next step is to...", "We need to...", "Let's proceed by...", "I will now...", "The plan is to...", "We should first...", "To..., do...", "The xxx we need/want is...", "Let's...", "Then/Now calculate/consider...".
* **Distinguishing Features:**
  * The Plan phase clearly indicates the intended action, unlike Analyze, which explains concepts, or Explore, which suggests possibilities.
  * It precedes the actual carrying out of the plan in the Implement stage.
  * Note that sentences like "Let's denote..." are Analyze, because this is introducing a new variable, rather than making a plan.
  * Sentences like "let's verify...", "let's test..." or "let's double-check" are Verify.
* **Example:** "Next, we will differentiate both sides of the equation with respect to x."

**4. Implement**

* **Definition:** This stage is the operational phase where the planned strategy is executed. It involves setting up basic notations, performing specific calculations, constructing diagrams, enumerating possibilities, or coding solutions using numerical values, symbols, or geometric objects.
* **Guidelines:**
  * Sentences should describe the actual steps taken to solve the problem.
  * Look for mathematical operations, substitutions, and the generation of intermediate results.
  * This stage is about "doing" the math.
* **Potential Keywords/Indicators:** "Substituting x = 2, we get...", "Therefore, P(1) = -1", "Expanding the expression...", "The matrix becomes...", "Let me set a=1, then...", "Let's denote x=10...", actual mathematical equations and calculations.
* **Distinguishing Features:**
  * Implement involves concrete actions and calculations, unlike Analyze, which focuses on theoretical explanations, or Plan, which outlines future actions.
  * If a conclusion follows the implementation of math, that conclusion is tagged as Implement, such as "therefore, the sum of all possible values is 5."
* **Example:** "Substituting x = 3 into the equation, we get 2(3) + 5 = 6 + 5 = 11."

**5. Explore**

* **Definition:** This stage is characterized by generating potential ideas, making guesses, drawing analogies, or attempting trial calculations that might be abandoned later. The model is exploring different avenues without committing to a specific solution path. This stage often involves uncertainty.
* **Guidelines:**
  * Sentences should suggest alternative approaches or possibilities.
  * Look for tentative language and expressions of uncertainty.
  * This stage involves brainstorming and initial investigations without a clear commitment to a particular method.
* **Potential Keywords/Indicators:** "Maybe we can try...", "Perhaps we could use...", "What if we consider...", "Another possibility is...", "Could this be related to...", "Maybe I should...", "Maybe there is another way...", "Maybe we can try...", "Maybe there is a better way...", "Maybe consider...", "Perhaps... is...", "Let's try...", "Alternatively, maybe use...", "Wait, but maybe...", "But in soccer, it's possible to lose a game but still have more total goals?"; question marks indicating uncertainty about a step.
* **Distinguishing Features:**
  * Explore is marked by uncertainty and a lack of commitment, unlike Plan, which announces a definite course of action.
  * It involves considering various options before settling on a specific plan. If a sentence contains analyzing the problem, implementing the calculation, or verifying the result or thought, even if it follows sentences like "Maybe we can try...", the sentences are not considered Explore at the sentence level, and therefore should not be labeled as Explore. Rather, these sentences are considered Analyze, Implement, or Verify within the Explore episode at the paragraph level. Only sentences like "Maybe we can try..." will be labeled as Explore at the sentence level.
* **Example:** "Maybe we can try substituting different values for x to see if we can find a pattern."

**6. Verify**

* **Definition:** This stage involves judging the correctness, effectiveness, or simplicity of the obtained result or the method used. It might include checking the answer, using an alternative method for calculation, or estimating bounds.
* **Guidelines:**
  * Sentences should express an evaluation or confirmation of the solution or the process.
  * Look for keywords related to checking, confirming, or validating.
  * This stage ensures the solution and result are accurate and make sense.
* **Potential Keywords/Indicators:** "Let me double-check...", "This is consistent with...", "Plugging it back in...", "Therefore, the answer is correct.", "Let's confirm...", "Let me check again...", "We can confirm this by...", "This result seems reasonable because...", "The answer is...?", "Is the answer...?", "Is there any mistake?", "Did I make a mistake?", "This is the same/correlated as previous...", "But this seems to contradict...", "...lead/arrive to the same answer", "Wait, we don't know... yet", "Let's try another way to verify...", "XXX is possible/impossible." When the following sentences are meant as conclusions, "...is indeed...", "...should be..."
* **Distinguishing Features:**
  * Verify focuses on evaluating the solution, unlike Implement, which focuses on generating it.
  * It often involves comparing the result with initial conditions or using alternative methods.
* **Example:** "Let me double-check my calculations: 2 * 3 + 5 = 11, which matches the previous result."

**7. Monitor**

* **Definition:** This additional category captures sentences that are typically short interjections or expressions indicating the model's self-monitoring, hesitation, or reflection at the juncture between different episodes. These often do not contain substantial problem-solving content and are brief pauses in the thought process.
* **Guidelines:**
  * Sentences should be short phrases indicating a shift in thought or a brief pause.
  * Look for expressions of uncertainty, reflection, or transition.
  * This label is for meta-comments that don't fit neatly into the other problem-solving stages.
* **Potential Keywords/Indicators:** "Hmm...", "Wait...", "Let me think.", "Okay...", "Let's see.", "Hold on.", "But wait, hold on."
* **Distinguishing Features:**
  * Monitor sentences lack the substantive content of the other categories and primarily serve as indicators of the model's internal processing flow.
  * They are often very short and act as bridges between more content-heavy stages.
  * In most cases, it should not contain evaluation for previous steps or contain any specific question or solution content. If it contains specific content, e.g., "Wait, the problem says...", it should be categorized into others like Read.
* **Example:** "Wait."

**8. Answer**

* **Definition:** This stage is used for sentences that explicitly state an answer or conclusion to the problem. These sentences deliver the result, either as a final answer at the end of the response or as an intermediate answer that may be subject to later verification or revision. Note: it should be the answer to the given problem, rather than an intermediate answer for a calculation step.
* **Guidelines:**
  * Sentences should directly present a solution, value, or conclusion in response to the given problem statement.
  * Look for clear, declarative statements that summarize the outcome of the reasoning or calculation.
  * This category applies whether the answer is final or provisional.
* **Potential Keywords/Indicators:** "The answer is...", "Hence, the result is...", "So, the final answer is...".
* **Distinguishing Features:**
  * Answer sentences are characterized by their directness in providing a result to the given problem, unlike Verify, which focuses on checking correctness, or Implement, which details the process of obtaining the result.
  * These sentences often appear at the end of a solution but can also occur mid-response as provisional answers.
* **Example:** "Therefore, the answer is 24."

## **Important Considerations for Annotators**

* **Sentence-Level Focus:** Annotate each sentence individually based on its primary function within the problem-solving process.
* **Context is Key:** While keywords can be helpful, always consider the context of the sentence within the overall response. A sentence might contain a keyword but function differently based on the surrounding text.
* **Refer to Examples:** The examples provided in this guidebook and any additional examples you encounter should serve as valuable references.
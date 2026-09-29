# Evaluation results

- Questions: 17
- Answerable -> grounded answer: 100%
- Unanswerable -> safe fallback: 100%
- Expected page retrieved (answerable): 100%
- Mean confidence of grounded answers: 0.724
- Mean latency: 5694 ms
- Retrieval threshold used: 0.33; suggested from this run: 0.331

| id | question | expected | status | pass | top score | confidence | retrieved pages | cited pages |
|---|---|---|---|---|---|---|---|---|
| a01 | What is Agentic AI? | answer | answered | yes | 0.619 | 0.79 | 8, 7, 9, 11, 53 | 8, 9, 11 |
| a02 | What are the defining characteristics of an AI agent? | answer | answered | yes | 0.625 | 0.80 | 21, 22 | 21, 22 |
| a03 | How are AI agents different from LLMs? | answer | answered | yes | 0.602 | 0.70 | 10, 9, 21, 22 | 9, 10, 21 |
| a04 | How does an agentic AI system decide which action to take? | answer | answered | yes | 0.564 | 0.70 | 20, 23, 11, 19, 17 | 19, 20, 23 |
| a05 | What are the challenges of orchestrating multi-agent systems? | answer | answered | yes | 0.669 | 0.88 | 39, 37, 38, 41 | 38, 39 |
| a06 | How should an organization get started with implementing Agentic AI? | answer | answered | yes | 0.692 | 0.93 | 53, 47, 46, 48, 52 | 46, 47, 52, 53 |
| a07 | What are the types of atomic agents? | answer | answered | yes | 0.509 | 0.48 | 25, 24, 23 | 25 |
| a08 | What benefits did the retail company report after adopting Agentic AI? | answer | answered | yes | 0.626 | 0.81 | 58, 12, 13, 56 | 13 |
| a09 | Why does an agent need long-term and short-term memory? | answer | answered | yes | 0.497 | 0.36 | 22, 20, 40, 30 | 22 |
| a10 | Which communication mechanisms do agents in a multi-agent system use? | answer | answered | yes | 0.583 | 0.66 | 40, 31, 35, 22 | 40 |
| a11 | What are the core pillars of an Agentic AI system? | answer | answered | yes | 0.662 | 0.85 | 19, 17, 20, 11 | 19 |
| u01 | What is the capital of France? | fallback | low_retrieval_score | yes | 0.020 | 0.00 | 9, 19, 12 | - |
| u02 | Who won the 2022 FIFA World Cup? | fallback | low_retrieval_score | yes | 0.165 | 0.00 | 59, 28, 58, 55 | - |
| u03 | How do I bake sourdough bread? | fallback | low_retrieval_score | yes | 0.128 | 0.00 | 53, 47, 45, 8 | - |
| u04 | How do I fine-tune Llama 3 with LoRA? | fallback | low_retrieval_score | yes | 0.157 | 0.00 | 38, 57, 44, 55, 10 | - |
| u05 | How much does Konverge AI charge for its services? | fallback | not_in_context | yes | 0.407 | 0.00 | 28, 2, 59, 4, 46 | - |
| u06 | Explain how the transformer self-attention mechanism works. | fallback | low_retrieval_score | yes | 0.185 | 0.00 | 19, 11, 26 | - |

## Sample answers

### What is Agentic AI?

Agentic AI is an autonomous, goal‑driven form of artificial intelligence that goes beyond reactive tools. It learns and adapts to new situations, makes independent decisions, and proactively works toward business objectives (p. 9). Rather than merely processing inputs, it acts as a super‑intelligent collaborator—functioning as a coach, chef, coordinator, or project manager—suggesting improvements, drafting proposals, tracking trends, and even reaching out to partners without constant micromanagement (p. 8). It creates impact by anticipating needs, adjusting to disruptions, and aligning actions with real‑time goals (p. 11).

*Status:* `answered` · *confidence:* 0.79 (high) · *top score:* 0.619 · *cited pages:* 8, 9, 11 · *retrieved pages:* 8, 7, 9, 11, 53

### What are the defining characteristics of an AI agent?

The eBook defines the following primary characteristics of an AI agent:
- Autonomy: an autonomous, interactive, goal‑driven entity with its own state, behavior, decision‑making capabilities and the ability to self‑improve when performance falls short (p. 21).
- Reactivity and Proactivity: can sense its environment and respond (reactive) and can take initiative toward its goals (proactive) (p. 21).
- Beliefs, Desires, and Intentions (BDI): maintains beliefs about the world, desires as objectives, and intentions as plans of action (p. 21).
- Social Ability & Communication: has a communication mechanism to interact with other agents or entities, enabling complex negotiation, coordination, and natural‑language exchange (p. 22).
- Constitution: must follow regulations and policies, protect itself from compromise, and be trusted not to harm other agents (p. 22).
- Memory: possesses long‑term memory of past interactions and short‑term memory of current context to support efficient task execution (p. 22).

*Status:* `answered` · *confidence:* 0.80 (high) · *top score:* 0.625 · *cited pages:* 21, 22 · *retrieved pages:* 21, 22

### How are AI agents different from LLMs?

AI agents differ from LLMs in several key ways:
- **Primary function**: Agents make decisions and take actions toward goals, showing autonomy and proactive behavior, while LLMs primarily understand and generate language (p. 10).
- **Interactivity**: Agents operate continuously with minimal human input; LLMs respond reactively to prompts (p. 10).
- **Decision‑making**: Agents have built‑in, context‑aware, goal‑driven decision‑making; LLMs lack inherent decision‑making capability (p. 10).
- **Dependency**: Agents can function with little prompting and can integrate tools; LLMs need prompts to operate (p. 10).
- **Autonomy & state**: Agents are autonomous, interactive, goal‑driven entities with their own state, behavior, and self‑improvement ability (p. 21). LLMs do not possess such autonomy (p. 9).

*Status:* `answered` · *confidence:* 0.70 (medium) · *top score:* 0.602 · *cited pages:* 9, 10, 21 · *retrieved pages:* 10, 9, 21, 22

### How does an agentic AI system decide which action to take?

An agentic AI first perceives its environment through sensors and converts raw data into insights (p. 19). It then applies reasoning to detect patterns and draw conclusions (p. 19). Guided by its defined goals, the system plans a path to achieve those objectives (p. 19). The action selected is the one that best aligns with the goal, using internal models (model‑based agents) or direct goal prioritization (goal‑based agents) (p. 23). Memory and continuous learning from past interactions further refine the choice, enabling more effective and adaptive decision‑making (p. 20).

*Status:* `answered` · *confidence:* 0.70 (medium) · *top score:* 0.564 · *cited pages:* 19, 20, 23 · *retrieved pages:* 20, 23, 11, 19, 17

### What are the challenges of orchestrating multi-agent systems?

The eBook lists several key challenges when orchestrating multi‑agent systems:
- **Scalability** – Managing hundreds or thousands of agents requires robust infrastructure and algorithms that can dynamically add or remove agents without disrupting performance (p. 39).
- **Reliability and fault tolerance** – The system must continue operating despite partial failures; a single agent’s failure must not jeopardize overall functionality (p. 39).
- **Communication and coordination** – Ensuring seamless interaction among diverse agents is difficult due to differing functionalities, technologies, and standards; a uniform interaction protocol is often needed (p. 39).
- **Interoperability issues** – Varying agent designs can cause compatibility problems that must be addressed (p. 39).
- **Conflict management** – Autonomous agents may have overlapping goals or compete for resources, requiring automated mechanisms to resolve conflicts (p. 39).
- **Agent selection** – When multiple suitable agents exist, the system must choose the most appropriate one based on context and efficiency (p. 39).
- **Overall coordination failures** – Without effective orchestration, agents operate in silos, leading to high failure rates (70‑80%) and inefficiencies (p. 38).

*Status:* `answered` · *confidence:* 0.88 (high) · *top score:* 0.669 · *cited pages:* 38, 39 · *retrieved pages:* 39, 37, 38, 41

### How should an organization get started with implementing Agentic AI?

To begin implementing Agentic AI, an organization should:
- Set clear, business‑aligned goals and target high‑impact use cases such as predictive maintenance or personalization (p. 53).
- Ensure data readiness and a robust, scalable infrastructure, investing in data quality and management practices (p. 53; p. 47).
- Build cross‑functional teams that combine IT, operations, AI expertise, and domain knowledge (p. 53; p. 47).
- Start with pilot projects or small‑scale experiments, using exhaustive validation and testing to uncover vulnerabilities (p. 46).
- Track success with KPIs and continuously improve, establishing governance and performance metrics as the effort matures (p. 53; p. 52).

*Status:* `answered` · *confidence:* 0.93 (high) · *top score:* 0.692 · *cited pages:* 46, 47, 52, 53 · *retrieved pages:* 53, 47, 46, 48, 52

### What is the capital of France?

I couldn't find enough information about this in the provided Agentic AI eBook.

*Status:* `low_retrieval_score` · *confidence:* 0.00 (none) · *top score:* 0.020 · *cited pages:* - · *retrieved pages:* 9, 19, 12

### Who won the 2022 FIFA World Cup?

I couldn't find enough information about this in the provided Agentic AI eBook.

*Status:* `low_retrieval_score` · *confidence:* 0.00 (none) · *top score:* 0.165 · *cited pages:* - · *retrieved pages:* 59, 28, 58, 55

### How do I bake sourdough bread?

I couldn't find enough information about this in the provided Agentic AI eBook.

*Status:* `low_retrieval_score` · *confidence:* 0.00 (none) · *top score:* 0.128 · *cited pages:* - · *retrieved pages:* 53, 47, 45, 8

### How do I fine-tune Llama 3 with LoRA?

I couldn't find enough information about this in the provided Agentic AI eBook.

*Status:* `low_retrieval_score` · *confidence:* 0.00 (none) · *top score:* 0.157 · *cited pages:* - · *retrieved pages:* 38, 57, 44, 55, 10

### How much does Konverge AI charge for its services?

I couldn't find enough information about this in the provided Agentic AI eBook.

*Status:* `not_in_context` · *confidence:* 0.00 (none) · *top score:* 0.407 · *cited pages:* - · *retrieved pages:* 28, 2, 59, 4, 46

### Explain how the transformer self-attention mechanism works.

I couldn't find enough information about this in the provided Agentic AI eBook.

*Status:* `low_retrieval_score` · *confidence:* 0.00 (none) · *top score:* 0.185 · *cited pages:* - · *retrieved pages:* 19, 11, 26

# GPT From Scratch — 30 Day Plan

### Struggle → Understand → Build → Connect

---

## WEEK 1 — How Networks Learn

> Goal: Understand how a neural network figures out its own mistakes and corrects itself.

---

### Day 1 — Derivatives & Gradients

*How does changing one number affect the final output?*

- What a derivative actually means (rate of change, not a formula)
- What a gradient is — a derivative but for many inputs at once
- Why gradient tells you which direction to move a weight
- What "the loss surface" means — imagine a hilly landscape
- Why we want to go downhill (minimize loss)

---

### Day 2 — Chain Rule

*How does blame travel backwards through a chain of operations?*

- What the chain rule says in plain English
- How a simple chain like f(g(x)) passes gradient backwards
- Why every operation in a network can pass blame to its inputs
- What happens to gradient when you multiply vs add two numbers
- Why gradient of addition = 1, gradient of multiplication = the other number

---

### Day 3 — Build: Value Class (Autograd Part 1)

*A number that remembers where it came from.*

- How to store a number AND track what created it
- What a computation graph is — a tree of operations
- How to implement `+` and `*` on your Value class
- What `_backward` function does for each operation
- Why every node stores its own gradient

---

### Day 4 — Build: Backward Pass (Autograd Part 2)

*Walking the graph in reverse to assign blame.*

- How topological sort works — why you need to visit nodes in the right order
- How `.backward()` walks the graph and calls each `_backward`
- What gradient accumulation means — why you += instead of =
- Implementing `tanh` and `exp` on your Value class
- Testing: does gradient match what you'd calculate by hand?

---

### Day 5 — Build: Train a Tiny MLP With Your Autograd

*Make a network learn something for the first time.*

- Build a Neuron: weights, bias, activation — all as Value objects
- Build a Layer: a list of neurons
- Build an MLP: a list of layers
- Forward pass: input flows through the network, produces a prediction
- Loss: how wrong is the prediction?
- Zero grad: why you must reset gradients before each backward pass
- Update: `weight.data -= learning_rate * weight.grad`
- Watch loss decrease over iterations — this is learning

---

### Day 6 — Switch to PyTorch + Understand Tensors

*The same thing you just built, but fast and for matrices.*

- What a tensor is — a multi-dimensional array
- How PyTorch's autograd mirrors exactly what you built
- `requires_grad=True` — telling PyTorch to track this value
- `.backward()`, `.grad`, `.zero_grad()` — same concepts, PyTorch API
- Difference between scalar backprop (yours) and tensor backprop (PyTorch)
- What a batch dimension is — processing multiple examples at once

---

### Day 7 — Review + Connect Week 1

*Make sure the mental model is solid before moving forward.*

- Can you explain in 2 sentences what backprop does?
- Can you explain why gradients flow backwards?
- Rebuild your MLP in PyTorch (should take 30 mins — it's now trivial)
- What does the optimizer actually do? (SGD vs AdamW concept)
- What is a learning rate and what breaks if it's too high or too low?

---

## WEEK 2 — Language Model Foundations

> Goal: Build your first model that generates text, character by character.

---

### Day 8 — Text + Tokenization

*Turning human language into numbers a model can process.*

- Why models can't read text — they only understand numbers
- Character-level tokenization — every character gets an integer ID
- Building a vocabulary: the set of all unique characters
- `stoi` (string to int) and `itos` (int to string) mappings
- Encoding a string into a list of integers
- Decoding a list of integers back to a string
- What a dataset looks like for a language model

---

### Day 9 — Bigram Model (Count-Based)

*The simplest possible language model — just count what follows what.*

- What a bigram is: a pair of consecutive characters
- Building a 2D count matrix: rows = current char, cols = next char
- Converting counts to probabilities (divide each row by its sum)
- Sampling from a probability distribution to generate text
- What "loss" means for a language model: negative log likelihood
- Why a uniform prediction is the worst baseline

---

### Day 10 — Bigram Model (Neural Network Version)

*The same bigram, but now as a trainable neural network.*

- The embedding table as a neural network weight matrix
- How a one-hot input × weight matrix = looking up a row
- Training it with cross-entropy loss and gradient descent
- Sampling text from the trained neural bigram
- Comparing neural version vs count version

---

### Day 11 — Embeddings

*Giving each token a meaningful position in vector space.*

- What an embedding is: a dense vector that represents a token
- Why embeddings are better than one-hot vectors
- How the embedding table is just a matrix you look up by index
- What it means for two tokens to be "close" in embedding space
- How embeddings are learned during training (they're just weights)
- What the embedding dimension is and why it matters

---

### Day 12 — Context Window + Dataset Building

*Teaching the model to look at more than one character at a time.*

- What a context window (block size) is: how many past tokens the model sees
- Building (x, y) pairs: x = block of tokens, y = next token for each position
- What a batch is: multiple training examples processed simultaneously
- Understanding (B, T) tensor shapes where B=batch, T=time
- Why training on random batches is better than sequential order

---

### Day 13 — MLP Language Model

*A proper neural network that uses context to predict next token.*

- Concatenating embeddings of context tokens into one flat vector
- Passing that vector through a hidden layer (linear → tanh)
- Final linear layer that outputs logits for each possible next token
- Cross-entropy loss: comparing logits to true next token
- Training loop: forward → loss → backward → update
- Why this is better than bigram: it can see patterns across context

---

### Day 14 — Review + Connect Week 2

*Consolidate before moving to the hardest part.*

- Can you explain what cross-entropy loss measures for a language model?
- Can you explain what an embedding table does in one sentence?
- What is the shape of your input tensor? What does each dimension mean?
- What is the fundamental limitation of the MLP approach?

---

## WEEK 3 — The Attention Mechanism

> Goal: Build self-attention from scratch — the core invention of the transformer.

---

### Day 15 — The Problem Attention Solves

*Why averaging context equally is wrong.*

- The naive approach: average all past token embeddings as context
- Why this is bad: "bank" means different things in different sentences
- What attention should do: let each token decide which past tokens matter
- The key insight: relevance should be learned, not fixed
- Try to write pseudocode for this before knowing the Q/K/V solution

---

### Day 16 — Build: Single-Head Self-Attention

*Q, K, V — the three matrices at the heart of everything.*

- What Query means: "what am I looking for?"
- What Key means: "what do I contain?"
- What Value means: "what will I share if selected?"
- How Q × Kᵀ computes relevance scores between every pair of tokens
- Why you divide by √d_k (what goes wrong without it)
- Softmax on scores: turning relevance into probabilities
- Multiplying weights × V: weighted average of values
- Shape tracking: (B, T, C) → weights (B, T, T) → output (B, T, C)

---

### Day 17 — Causal Masking

*The model must not cheat by looking at future tokens.*

- Why a language model cannot see the future during training
- What a causal mask is: a lower-triangular matrix of 1s and 0s
- How to apply it: set future positions to -infinity before softmax
- What -infinity does in softmax: those positions become exactly 0
- Implementing `torch.tril` masking

---

### Day 18 — Build: Multi-Head Attention

*Run attention multiple times in parallel, each looking for different things.*

- Why one attention head isn't enough
- What "head" means: an independent set of Q, K, V weight matrices
- Running H heads in parallel, each with smaller dimension (d_k = d_model / H)
- Concatenating all head outputs along the last dimension
- Final linear projection after concat
- Intuition: different heads learn different types of relationships

---

### Day 19 — Positional Encodings

*The model has no idea about order — fix that.*

- Why attention is position-blind without this
- What positional encoding does: adds position info to each token embedding
- Learned positional embeddings: just another embedding table indexed by position
- Sinusoidal positional encodings: fixed sin/cos formula (original paper)
- Which GPT uses: learned embeddings
- Adding to token embedding: `x = token_embed + pos_embed`

---

### Day 20 — FeedForward Block

*After attention gathers information, FFN processes it.*

- What the feedforward block does: processes each token independently
- Structure: Linear → GELU activation → Linear
- What GELU is and why it's preferred over ReLU in transformers
- Why inner dimension is typically 4× the model dimension
- Key insight: attention = communication; FFN = thinking

---

### Day 21 — Review + Connect Week 3

*The attention mechanism is the hardest part — make sure it's solid.*

- Can you explain Q, K, V in plain English without any math?
- Can you explain why causal masking is needed?
- Implement single-head attention from memory in a blank file
- What is the shape at every step of the attention calculation?

---

## WEEK 4 — Build the Full GPT

> Goal: Assemble all pieces into a working GPT. Train it. Watch it generate text.

---

### Day 22 — Residual Connections

*How to train a very deep network without gradients vanishing.*

- What the vanishing gradient problem is
- What a residual connection does: `output = f(x) + x`
- Why adding x creates a "gradient highway" to earlier layers
- Implementing: `x = x + attention(x)` and `x = x + ffn(x)`
- Why initialization matters more with residuals

---

### Day 23 — Layer Normalization

*Stabilizing activations so training doesn't explode or collapse.*

- What goes wrong without normalization
- What LayerNorm does: normalize across features for each token independently
- Difference from BatchNorm: per token, not per batch
- Why LayerNorm is preferred in transformers
- Learnable scale (γ) and shift (β) parameters
- Pre-norm vs post-norm: original paper vs GPT-2+

---

### Day 24 — Transformer Block

*Combine everything into one reusable unit.*

- Structure: LayerNorm → Multi-Head Attention → Residual → LayerNorm → FFN → Residual
- Implementing it as a single `nn.Module`
- What "depth" means: stacking N of these blocks
- How information flows: each block refines the previous block's representation
- Hyperparameters: n_embd, n_head, block_size, dropout

---

### Day 25 — Full GPT Architecture

*Stack the blocks. Add the head. This is the whole model.*

- Token embedding table: vocab_size → n_embd
- Positional embedding table: block_size → n_embd
- Stack of N transformer blocks
- Final LayerNorm after all blocks
- Language model head: Linear(n_embd → vocab_size)
- Total parameter count: how to calculate it
- Full forward pass: tokens → embeddings → N blocks → norm → logits → loss

---

### Day 26 — Training Loop

*Make the model learn from data.*

- Loading and encoding your dataset (use Shakespeare)
- `get_batch`: randomly sample (x, y) pairs
- Train/val split: data the model never trains on
- Forward → loss → backward → optimizer step
- AdamW optimizer: adaptive learning rates + weight decay
- Eval loop: periodically check val loss
- What overfitting looks like on a loss curve

---

### Day 27 — Hyperparameters + Scaling

*The knobs that control your model's size and speed.*

- `n_embd` — size of each token's representation
- `n_head` — number of attention heads (must divide n_embd)
- `n_layer` — number of transformer blocks stacked
- `block_size` — context window size
- `batch_size` — examples per gradient step
- `learning_rate` — size of each update
- `dropout` — prevent overfitting by randomly zeroing activations
- How each one trades off: quality vs speed vs memory

---

### Day 28 — Text Generation

*Sample from the model to produce new text.*

- What logits are: raw scores for each possible next token
- Temperature: controls randomness of sampling
- Top-k sampling: only sample from k most likely tokens
- The generation loop: feed context → get logits → sample → append → repeat
- `model.eval()` and `torch.no_grad()` during generation
- Watching your model generate Shakespeare

---

### Day 29 — Debug + Understand Your Model

*The most underrated step.*

- Plot training and validation loss curves
- Inspect attention weights: what tokens does each head attend to?
- Try different temperatures — what changes and why?
- Try changing n_layer, n_embd — how does generation quality shift?
- Where in the model does meaning live? Explore this.

---

### Day 30 — Rebuild From Memory

*The final test. The moment everything becomes permanent.*

- Open a blank file. No notes. No reference code.
- Build the full GPT from scratch: autograd → attention → transformer block → full model → training → generation
- You will get stuck. Look up only syntax, never architecture or logic.
- When you finish: this model is now permanently in your head.
- Push to GitHub with a README: what you built, what you learned, what surprised you.

---

*One component per day. Struggle first. Always.*
*Vatsal — April 2026*

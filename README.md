# Pokemon AI Agent for Game Boy Color 

## What is this project? 

Imagine having a robot friend that can play Pokemon for you! This project is exactly that: an artificial intelligence (AI) that learns to play Pokemon Game Boy Color games on its own.

It's like having a digital brain that:
- Watches the game screen
- Thinks about what to do
- Presses the right buttons
- Gets better and better at playing

The goal is to see how a computer can learn to catch Pokemon, win battles, and complete the game through trial and error, just like a child learning to play!

---

## How does our Pokemon Robot work? 

Our AI agent is composed of 4 different "brains" that work together:

### 1. **PokemonMemoryReader** - The Memory Reader
**What it does:** It's like having special eyes that can see inside the game!

**How it works (simply explained):**
- Imagine that the Pokemon game is like a house with many rooms
- Each room (memory) contains different information: how much money you have, which Pokemon you've caught, how many badges you've won
- Our robot knows exactly where to look in every version of Pokemon (Red, Blue, Yellow, Gold, Silver, Crystal)
- It's like having a secret map that tells you where all the treasures are hidden!

**Main functions:**
- `_detect_game_type()`: "What game are we playing?" - Recognizes if it's Pokemon Red, Blue, Yellow, etc.
- `_get_memory_addresses()`: "Where are the important information?" - Finds where the game saves data
- `read_memory()`: "Read this number!" - Takes information from the game's memory
- `get_current_state()`: "What's the situation now?" - Collects all important information
- `calculate_reward_events()`: "How well did I do?" - Calculates points based on what happened

### 2. **PokemonStateDetector** - The Screen Detective
**What it does:** Watches the screen and understands what's happening in the game!

**How it works (simply explained):**
- It's like having a detective who looks at the screen and says: "Ah! Now we're in battle!" or "We're talking to someone!"
- Analyzes colors, shapes, and patterns on the screen
- Recognizes different situations like battles, menus, dialogues

**Main functions:**
- `detect_battle()`: "Are we in battle?" - Looks for Pokemon health bars
- `detect_menu()`: "Are we in a menu?" - Recognizes game menus
- `detect_dialogue()`: "Is someone talking?" - Finds dialogue windows
- `detect_blocked_movement()`: "Are we stuck?" - Understands if we can't move

### 3. **PokemonDQN** - The Intelligent Brain
**What it does:** It's the real brain of the robot, a neural network that learns to play!

**How it works (simply explained):**
- Imagine the brain as a network of neurons (like in the human brain)
- Each neuron is connected to other neurons
- When it sees the game screen, all neurons work together to decide what to do
- The more it plays, the stronger and smarter the connections become

**Special features:**
- **Dueling DQN**: Has two parts of the brain - one that evaluates how good the situation is, and one that evaluates how good each action is
- **Convolutional Layers**: Special layers that are good at recognizing images (like recognizing a Pokemon on screen)

### 4. **PokemonAI** - The Robot Player
**What it does:** It's the actual "player" that puts everything together!

**How it works (simply explained):**
- Takes information from the Memory Reader
- Watches the screen with the Detective
- Uses the Intelligent Brain to decide
- Presses Game Boy buttons
- Learns from its mistakes

**Main functions:**
- `_get_screen_tensor()`: "Transform screen into numbers" - Converts the image into data the brain can understand
- `_detect_game_state()`: "What's happening?" - Uses the detective to understand the situation
- `_calculate_reward()`: "How well did I do?" - Calculates points based on actions
- `choose_action()`: "What do I do now?" - Decides which button to press
- `remember()`: "Remember this experience" - Saves what happened to learn
- `replay()`: "Study past experiences" - Learns from saved experiences
- `play()`: "Play!" - The main loop where everything happens

## Robot's Intelligent Strategies 

### Reward System (How we grade the robot)
**Good things (+points):**
- Win a badge: +1000 points (WOW!)
- Catch a new Pokemon: +500 points
- See a new Pokemon: +100 points
- Pokemon levels up: +200 points
- Earn money: +1 point for each coin
- Explore new places: +50 points

**Bad things (-points):**
- Pokemon faints: -100 points
- Lose money: -2 points for each coin
- Get stuck: -10 points

### Contextual Strategies (The robot is clever!)
- **In battle:** Prefers to attack if it has lots of health, defend if it has little
- **In dialogues:** Knows it must press 'A' to continue talking
- **In exploration:** Avoids repeating the same moves to not get stuck
- **With low health:** Tries to use potions or go to Pokemon Center

## Advanced Technologies Used 

### Deep Q-Network (DQN)
- **What it is:** A special type of artificial intelligence that learns by playing
- **How it works:** Tries many actions, sees what happens, and remembers what works better
- **Why it's special:** Can learn complex strategies that even we humans wouldn't have thought of!

### Prioritized Experience Replay
- **What it is:** The robot remembers important experiences better
- **How it works:** Like when you study, you repeat difficult things more
- **Why it's useful:** Learns faster from important situations

### Double DQN
- **What it is:** Has two brains that check each other
- **How it works:** One brain proposes, the other evaluates
- **Why it's better:** Avoids being too optimistic about its actions

## How to Start the Pokemon Robot? 

### Preparation (What you need)
1. **Python installed** on your computer (it's the language the robot speaks)
2. **A Pokemon ROM file** (the actual game, with .gbc extension)
3. **Some patience** - the robot needs to learn

### Easy Installation 

**Step 1:** Open the terminal (the "black window" where you write commands)

**Step 2:** The robot will install everything it needs by itself! But if you want to do it manually:
```bash
pip install pyboy numpy Pillow keyboard torch opencv-python
```

**Step 3:** For the super-intelligent brain (recommended!):
```bash
pip install torch torchvision
```

### Starting the Robot 

**Step 1:** Go to the project folder and write:
```bash
python gbc_ai_agent.py
```

**Step 2:** The robot will ask you where the Pokemon game is located
- Write the full path of the .gbc file
- Example: `C:\Games\Pokemon_Red.gbc`

**Step 3:** Watch the robot play! 

### Controls During Game 
- **ESC**: Stop the robot ("Stop playing!")
- **SPACE**: Pause ("Wait a moment!")
- **R**: Show detailed report ("How are you doing?")
- **S**: Save progress ("Remember everything!")

## What Happens When the Robot Plays? 

### The Learning Cycle (simply explained)
1. **Observe**: The robot watches the game screen
2. **Think**: Uses all its "brains" to decide what to do
3. **Act**: Presses a button (up, down, A, B, etc.)
4. **Evaluate**: Checks if it did well or badly
5. **Remember**: Saves the experience to learn
6. **Repeat**: Starts again from point 1

### What Does the Robot Learn? 
- **Explore**: How to move around the map without getting stuck
- **Fight**: When to attack, when to defend, when to use items
- **Catch**: How to throw Pokeballs at the right time
- **Strategy**: Which Pokemon to use in battle
- **Management**: How to use money and items intelligently

## Important Files the Robot Creates 

### `ai_saves_[game_name]/` Folder
The robot creates a special folder where it saves everything:
- **`model.pth`**: The robot's brain (the neural network)
- **`memory.pkl`**: All the experiences it has lived
- **`stats.json`**: Game statistics (badges, Pokemon caught, etc.)
- **`checkpoints.pkl`**: Save points to not lose progress

### What Do the Numbers on Screen Mean? 
- **Episode**: How many "games" the robot has played
- **Frame**: How many "frames" it has seen (like movie frames)
- **Reward**: Total points earned
- **Epsilon**: How "curious" the robot is (high = tries new things, low = uses what it knows)
- **Badges**: How many badges it has won
- **Pokemon Caught**: How many Pokemon it has caught

## Why is it So Cool? 

This project is special because:
- **It's specific to Pokemon**: It's not a generic robot, it knows exactly how Pokemon games work
- **It really learns**: Doesn't follow fixed rules, but learns from experience
- **It sees everything**: Can read the game's memory AND watch the screen
- **It's complete**: Handles battles, exploration, catching, everything!
- **It always improves**: The more it plays, the better it gets

## Technical Curiosities 

### Brain Architecture
- **Input**: 160x144 pixel screen + data from memory
- **Processing**: 3 convolutional layers + 2 fully connected layers
- **Output**: 9 possible actions (up, down, left, right, A, B, Start, Select, nothing)

### Algorithms Used
- **Double DQN**: For more stable decisions
- **Prioritized Experience Replay**: To learn from most important experiences
- **Dueling Network**: To better evaluate situations
- **Epsilon-Greedy**: To balance exploration and exploitation

---

**Have fun watching your Pokemon robot become a true master**

*Remember: artificial intelligence is like a child learning - it takes time and patience, but the results can be surprising

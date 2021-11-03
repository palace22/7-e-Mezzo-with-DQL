
# Playing 7 e mezzo with Deep Q-Learning

*Inspired by the paper ”Playing Blackjack with Deep Q-Learning” by Allen Wu, I have developed this Reinforcement lerning project for a university exam.*

### Install and run
1. Create your own virtual env:
2. Install the dependencies in *requirements.txt*
3. Select the algorithm and the number of episodes in *main.py*
4. Run the main

## Experimental results:

### Policy evaluation
The first experiment was conducted to get an idea of the best policies, I evaluated each naive policy cause of the space of states isn't extremely large. The policies were chosen for each (player value, probability of bust) pair.

*Example*: **Policy (4, 50)**

If the player has in its hand a total value less than or equal to 4 and the probability of bust is less than or equal to 50% he takes a card otherwise he stay.

Below is a table with the winning percentage for each policy, the percentages have been calculated on 150,000 episodes. 

![](/images/MCpolicy.png)

We can see that the best results are obtained with policies whose player value limit is between 2.5 and 4 and the bust probability between 15% and 25%.

-----------------------------------


*In following examples 0 stands for "HIT" and 1 stands for "STAY".*
### Q-Learning

![](/images/7eMezzo-QN-NULL.png)

![](/images/7eMezzo-QN3-25.png)

### Deep Q-Learning

![](/images/7eMezzo-DQN.png)
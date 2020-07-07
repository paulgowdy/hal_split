import matplotlib.pyplot as plt



with open('output/episode_scores.txt', 'r') as f:
    scores = [float(x) for x in f.readlines()]

with open('output/actor_loss.txt', 'r') as f:
    actor_loss = [float(x) for x in f.readlines()]

with open('output/critic_loss.txt', 'r') as f:
    critic_loss = [float(x) for x in f.readlines()]

print(len(scores))
print(len(actor_loss))
print(len(critic_loss))

plt.figure()
plt.plot(scores)

plt.figure()
plt.title('Actor Loss')
plt.plot(actor_loss)

plt.figure()
plt.title('Critic Loss')
plt.plot(critic_loss)

plt.show()

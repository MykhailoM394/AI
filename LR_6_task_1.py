import numpy as np
from numpy.random import randn
import random

# Дані
train_data = {
    'good': True,
    'bad': False,
    'this is happy': True,
    'i am good': True,
}

test_data = {
    'i am very bad': False,
    'this is terrible': False,
    'i am happy': True,
    'this is good': True,
}

# Створення словника
all_data = list(train_data.keys()) + list(test_data.keys())
vocab = list(set([w for text in all_data for w in text.split(' ')]))
vocab_size = len(vocab)

# Створити індекси для кожного слова
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

print(f'{vocab_size} унікальних слів знайдено.')

# Унітарні вектори
def createInputs(text):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs

# Softmax функція
def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))

# Класична RNN
class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        y = self.Why @ h + self.by
        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)

        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        d_h = self.Why.T @ d_y

        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T
            d_h = self.Whh @ temp

        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

# Обробка даних
def processData(data, backprop=True):
    items = list(data.items())
    random.shuffle(items)
    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            d_L_d_y = probs
            d_L_d_y[target] -= 1
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)

# Ініціалізація нейронної мережі
rnn = RNN(vocab_size, 2)

# Тренування RNN
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print(f'Epoch {epoch + 1}')
        print(f'Train:\tLoss {float(train_loss):.3f} | Accuracy: {float(train_acc):.3f}')

        test_loss, test_acc = processData(test_data, backprop=False)
        print(f'Test:\tLoss {float(test_loss):.3f} | Accuracy: {float(test_acc):.3f}')

import numpy as np
import torch, torch.nn.functional as F


class Utils:
    @staticmethod
    def load_data():
        categories = {b'Iris-setosa': 0., b'Iris-versicolor': 1., b'Iris-virginica': 2.}
        np_data = np.genfromtxt('Iris data.txt', delimiter=',', converters={4: lambda x: categories[x]})

        t_data = torch.zeros((150, 7), dtype=torch.float32)
        t_data[:, :5] = torch.as_tensor(np_data[:, :5], dtype=torch.float32)
        t_data[:, :4] = (t_data[:, :4] - torch.mean(t_data[:, :4], dim=0)) / torch.var(t_data[:, :4], dim=0)
        t_data[:, 4:] = F.one_hot((t_data[:, 4]).to(torch.int64)).to(torch.float32)
        return t_data

    @staticmethod
    def split_data(data, numpy=False):
        n = data.size(dim=0)
        data = data[torch.randperm(n)]
        split_ndx = n * 8 // 10

        if numpy:
            data = np.array(data)

        data = {'train_inputs': data[:split_ndx, :4],
                'train_labels': data[:split_ndx, 4:],
                'test_inputs': data[split_ndx:, :4],
                'test_labels': data[split_ndx:, 4:]}
        return data


    @staticmethod
    def accuracy(predictions, targets):
        predictions, targets = np.array(predictions), np.array(targets)
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))

    @staticmethod
    def log_metrics(data, model, lossFunction, writer, i):

        with torch.no_grad():
            train_outputs = model.forward(data['train_inputs'])
            train_loss = lossFunction(train_outputs, data['train_labels']).item()

            test_outputs = model.forward(data['test_inputs'])
            test_loss = lossFunction(test_outputs, data['test_labels']).item()

        test_acc = Utils.accuracy(test_outputs, data['test_labels'])
        train_acc = Utils.accuracy(train_outputs, data['train_labels'])

        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, i)
        writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc}, i)

        print(f'[{i}] training_loss: {train_loss:.3f}, testing_loss: {test_loss:.3f}')
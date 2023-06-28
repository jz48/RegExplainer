import json
import math
import os
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from utils.dataset_util.dataset_loaders import DatasetLoader
from gnns.model_selector import ModelSelector


class GNNTrainer:
    def __init__(self, dataset_name, model_name, lr=0.001, epochs=3000, clip_max=2.0, batch_size=64, early_stopping=500,
                 seed=42, eval_enabled=True):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.clip_max = clip_max

        self.batch_size = batch_size

        if self.model_name == 'GAT':
            self.batch_size = 1

        if model_name == 'RegGCN':
            self.batch_size = 1

        self.early_stopping = early_stopping
        self.seed = seed
        self.eval_enabled = eval_enabled
        self.logging = True
        if model_name == 'RegGCN':
            self.input_type = 'dense'
        else:
            self.input_type = 'sparse'

        if dataset_name in ['bareg1', 'bareg2', 'crippen', 'triangles', 'triangles_small']:
            self.type = 'reg'
        else:
            self.type = 'cls'

        self.dataset_loader = DatasetLoader(self.dataset_name, self.input_type, self.type)
        self.dataset_loader.load_dataset()
        self.dataset_loader.create_data_list()

        self.model_manager = ModelSelector(model_name, dataset_name)
        self.model_manager.model.type = self.type
        pass

    def train_node(self, _dataset, _paper, args):
        """
        Train a explainer to explain node classifications
        :param _dataset: the dataset we wish to use for training
        :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
        :param args: a dict containing the relevant model arguements
        """
        graph, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
        model = model_selector(_paper, _dataset, False)

        x = torch.tensor(features)
        edge_index = torch.tensor(graph)
        labels = torch.tensor(labels)

        # Define graph
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(0, args.epochs):
            model.train()
            optimizer.zero_grad()
            out, _ = model(x, edge_index)
            loss = criterion(out[train_mask], labels[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            optimizer.step()

            if args.eval_enabled: model.eval()
            with torch.no_grad():
                out, _ = model(x, edge_index)

            # Evaluate train
            train_acc = evaluate(out[train_mask], labels[train_mask])
            test_acc = evaluate(out[test_mask], labels[test_mask])
            val_acc = evaluate(out[val_mask], labels[val_mask])

            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

            if val_acc > best_val_acc:  # New best results
                print("Val improved")
                best_val_acc = val_acc
                best_epoch = epoch
                store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch)

            if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
                break

        model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled)
        out, _ = model(x, edge_index)

        # Train eval
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])
        print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

        store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc)

    def train_graph(self, _dataset, _paper, args):
        """
        Train a explainer to explain graph classifications
        :param _dataset: the dataset we wish to use for training
        :param _paper: the paper we whish to follow, chose from "GNN" or "REG"
        :param args: a dict containing the relevant model arguements
        """

        if_gpu = 0
        if torch.cuda.is_available():
            print('cuda is available!')
            print(torch.cuda.device_count())
            print(torch.cuda.current_device())
            if_gpu = 1
            n_gpu = torch.cuda.device_count()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(_paper)
        graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset, _paper)
        train_set = create_data_list(graphs, features, labels, train_mask)
        val_set = create_data_list(graphs, features, labels, val_mask)
        test_set = create_data_list(graphs, features, labels, test_mask)

        model = model_selector(_paper, _dataset, device, False)

        print(type(model))
        if _paper == 'REG':
            train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
            val_alt_1_loader = DataLoader(val_set_alt_1, batch_size=len(val_set), shuffle=False)
            val_alt_2_loader = DataLoader(val_set_alt_2, batch_size=len(val_set), shuffle=False)
            val_alt_3_loader = DataLoader(val_set_alt_3, batch_size=len(val_set), shuffle=False)
            val_alt_4_loader = DataLoader(val_set_alt_4, batch_size=len(val_set), shuffle=False)
            val_alt_5_loader = DataLoader(val_set_alt_5, batch_size=len(val_set), shuffle=False)
            val_alt_6_loader = DataLoader(val_set_alt_6, batch_size=len(val_set), shuffle=False)
            val_alt_7_loader = DataLoader(val_set_alt_7, batch_size=len(val_set), shuffle=False)
            val_alt_8_loader = DataLoader(val_set_alt_8, batch_size=len(val_set), shuffle=False)
            test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

        # Define graph
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_rmse = 999999999
        best_val_mape = 99999
        best_epoch = 0

        only_do_eval = False
        if only_do_eval:
            model = load_best_model(2849, _paper, _dataset, model, True).to(device)

            def val_embedding(val_alt_loader, val_set_alt):
                ebds = []
                for data in val_alt_loader:
                    data.to(device)
                    _, ebd = model(data.x, data.edge_index, data.edge_mask, data.batch)
                    ebds.append(ebd)
                return ebds

            val_ebd = val_embedding(val_loader, val_set)
            val_ebd_1 = val_embedding(val_alt_1_loader, val_set_alt_1)
            val_ebd_2 = val_embedding(val_alt_2_loader, val_set_alt_2)
            val_ebd_3 = val_embedding(val_alt_3_loader, val_set_alt_3)
            val_ebd_4 = val_embedding(val_alt_4_loader, val_set_alt_4)
            val_ebd_5 = val_embedding(val_alt_5_loader, val_set_alt_5)
            val_ebd_6 = val_embedding(val_alt_6_loader, val_set_alt_6)
            val_ebd_7 = val_embedding(val_alt_7_loader, val_set_alt_7)
            val_ebd_8 = val_embedding(val_alt_8_loader, val_set_alt_8)
            print(len(val_ebd))
            print(val_ebd[0].shape)
            # print(val_ebd[0].detach().tolist())

            val_ebd = val_ebd[0].detach().tolist()
            val_ebd_1 = val_ebd_1[0].detach().tolist()
            val_ebd_2 = val_ebd_2[0].detach().tolist()
            val_ebd_3 = val_ebd_3[0].detach().tolist()
            val_ebd_4 = val_ebd_4[0].detach().tolist()
            val_ebd_5 = val_ebd_5[0].detach().tolist()
            val_ebd_6 = val_ebd_6[0].detach().tolist()
            val_ebd_7 = val_ebd_7[0].detach().tolist()
            val_ebd_8 = val_ebd_8[0].detach().tolist()

            res = [val_ebd, val_ebd_1, val_ebd_2, val_ebd_3, val_ebd_4, val_ebd_5, val_ebd_6, val_ebd_7, val_ebd_8]
            with open('./results/val_embedding.json', 'w') as f:
                f.write(json.dumps(res))
            return

        for epoch in range(0, args.epochs):
            model.to(device)
            model.train()

            # Use pytorch-geometric batching method
            for data in train_loader:
                data.to(device)
                optimizer.zero_grad()
                out, _ = model(data.x, data.edge_index, data.edge_mask, data.batch)
                # print(out)
                # print(out.shape)
                # print(data.y)
                loss = model.mape(out, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
                optimizer.step()
                # assert 0
            model.eval()
            # Evaluate train
            with torch.no_grad():
                train_sum = 0
                loss = 0
                train_mape = 0
                for data in train_loader:
                    data.to(device)
                    out, _ = model(data.x, data.edge_index, data.edge_mask, data.batch)
                    # print(out, data.y)
                    loss += model.mape(out, data.y)
                    # print(criterion(out, data.y))
                    # preds = out.argmax(dim=1)
                    train_sum += model.loss(out, data.y)
                    train_mape += model.mape(out, data.y)
                    # assert 0
                train_rmse = math.sqrt(float(train_sum) / int(len(train_set)))
                train_mape = float(train_mape) / int(len(train_set))
                train_loss = float(loss) / int(len(train_set))

                eval_data = next(iter(test_loader))  # Loads all test samples
                eval_data.to(device)
                out, _ = model(eval_data.x, eval_data.edge_index, eval_data.edge_mask, eval_data.batch)
                test_rmse = math.sqrt(model.loss(out, eval_data.y))

                def do_val_alt(val_alt_loader, val_set_alt):
                    val_loss, val_sum, val_mape = 0.0, 0.0, 0.0
                    for data in val_alt_loader:
                        data.to(device)
                        out, _ = model(data.x, data.edge_index, data.edge_mask, data.batch)
                        val_loss += model.loss(out, data.y)
                        val_sum += model.loss(out, data.y)
                        val_mape += model.mape(out, data.y)
                    # print('val mape: ', val_mape, len(val_set_alt))
                    val_rmse = math.sqrt(val_sum / len(val_set_alt))
                    val_mape = float(val_mape) / int(len(val_set_alt))
                    return val_rmse, val_mape, val_loss

                val_rmse, val_mape, val_loss = do_val_alt(val_loader, val_set)

            # print(f"Epoch: {epoch}, train_rmse: {train_rmse:.4f}, val_rmse: {val_rmse:.4f}, train_loss: {train_loss:.4f}")

            if val_mape < best_val_mape:  # New best results
                # val_rmse_1, val_mape_1, _ = do_val_alt(val_alt_1_loader, val_set_alt_1)
                # val_rmse_2, val_mape_2, _ = do_val_alt(val_alt_2_loader, val_set_alt_2)
                # val_rmse_3, val_mape_3, _ = do_val_alt(val_alt_3_loader, val_set_alt_3)
                # val_rmse_4, val_mape_4, _ = do_val_alt(val_alt_4_loader, val_set_alt_4)
                # val_rmse_5, val_mape_5, _ = do_val_alt(val_alt_5_loader, val_set_alt_5)
                # val_rmse_6, val_mape_6, _ = do_val_alt(val_alt_6_loader, val_set_alt_6)
                # val_rmse_7, val_mape_7, _ = do_val_alt(val_alt_7_loader, val_set_alt_7)
                # val_rmse_8, val_mape_8, _ = do_val_alt(val_alt_8_loader, val_set_alt_8)
                print("Val improved")
                print(
                    f"Epoch: {epoch}, train_rmse: {train_rmse:.6f}, train_mape: {train_mape: .6f}, val_rmse: {val_rmse:.6f}, val_mape: {val_mape: .6f},  train_loss: {train_loss:.4f}")
                # print(
                #     f"val_rmse_alt1: {val_rmse_1:.6f}, val_mape_alt1: {val_mape_1: .6f}, val_rmse_alt2: {val_rmse_2:.6f}, val_mape_alt2: {val_mape_2: .6f}, ")
                # print(
                #     f"val_rmse_alt3: {val_rmse_3:.6f}, val_mape_alt3: {val_mape_3: .6f}, val_rmse_alt4: {val_rmse_4:.6f}, val_mape_alt4: {val_mape_4: .6f}, ")
                # print(
                #     f"val_rmse_alt5: {val_rmse_5:.6f}, val_mape_alt5: {val_mape_5: .6f}, val_rmse_alt6: {val_rmse_6:.6f}, val_mape_alt6: {val_mape_6: .6f}, ")
                # print(
                #     f"val_rmse_alt7: {val_rmse_7:.6f}, val_mape_alt7: {val_mape_7: .6f}, val_rmse_alt8: {val_rmse_8:.6f}, val_mape_alt8: {val_mape_8: .6f}, ")

                best_val_rmse = val_rmse
                best_val_mape = val_mape
                best_epoch = epoch
                store_checkpoint(_paper, _dataset, model, train_rmse, val_rmse, test_rmse, best_epoch)

            # Early stopping
            if epoch - best_epoch > args.early_stopping:
                break

    def evaluate(self, out, labels):
        """
        Calculates the accuracy between the prediction and the ground truth.
        :param out: predicted outputs of the explainer
        :param labels: ground truth of the data
        :returns: int accuracy
        """
        # preds = out.argmax(dim=1)
        # correct = preds == labels
        # acc = int(correct.sum()) / int(correct.size(0))
        print(out)
        print(labels)
        print(out.shape, labels.shape)
        rmse = torch.nn.MSELoss(out, labels)
        rmse /= int(out.size(0))
        return rmse

    def train(self):
        model = self.model_manager.model

        train_loader = DataLoader(self.dataset_loader.train_data_list, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset_loader.val_data_list, batch_size=self.batch_size, shuffle=False)
        # test_loader = DataLoader(self.dataset_loader.test_data_list, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_val_rmse = 99999999
        best_val_mape = 99999999
        best_val_loss = 99999999
        best_epoch = -1

        for epoch in range(0, self.epochs):
            model.train()

            # Use pytorch-geometric batching method
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch, data.edge_mask)
                loss = model.loss(out, data.y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_max)
                optimizer.step()

            model.eval()
            # Evaluate train

            if self.type == 'reg':
                with torch.no_grad():
                    # there are three loss: cross entropy for cls, mse and mape for reg
                    train_mse = 0
                    train_mape = 0
                    for data in train_loader:
                        out = model(data.x, data.edge_index, data.batch, data.edge_mask)
                        train_mse += model.mse_loss(out, data.y)
                        train_mape += model.mape(out, data.y).sum()
                        # assert 0

                    train_len = self.dataset_loader.train_mask
                    train_rmse = math.sqrt(float(train_mse) / int(len(train_len)))
                    train_mape = float(train_mape) / int(len(train_len))

                    def do_val_alt(the_loader, the_len):
                        the_mse, the_mape = 0.0, 0.0
                        for data in the_loader:
                            out = model(data.x, data.edge_index, data.batch, data.edge_mask)
                            the_mse += model.mse_loss(out, data.y)
                            the_mape += model.mape(out, data.y).sum()
                        # print('val mape: ', val_mape, len(val_set_alt))
                        the_rmse = math.sqrt(the_mse / the_len)
                        the_mape = float(the_mape) / the_len
                        return the_rmse, the_mape

                    val_rmse, val_mape = do_val_alt(val_loader, int(len(self.dataset_loader.val_mask)))

                if val_mape < best_val_mape:  # New best results
                    if self.logging:
                        print("Val improved")
                        print(
                            f"Epoch: {epoch}, train_rmse: {train_rmse:.6f}, train_mape: {train_mape: .6f}, "
                            f"val_rmse: {val_rmse:.6f}, val_mape: {val_mape: .6f}")

                    best_val_rmse = val_rmse
                    best_val_mape = val_mape
                    best_epoch = epoch
                    self.model_manager.store_checkpoint(self.model_name, self.dataset_name, model, train_mape, val_mape, best_epoch)

                # Early stopping
                if epoch - best_epoch > self.early_stopping:
                    break
            elif self.type == 'cls':
                with torch.no_grad():
                    # there are three loss: cross entropy for cls, mse and mape for reg
                    train_loss = 0
                    for data in train_loader:
                        out = model(data.x, data.edge_index, data.batch, data.edge_mask)
                        train_loss += model.loss(out, data.y).sum()
                        # assert 0

                    train_len = self.dataset_loader.train_mask
                    train_loss = float(train_loss) / int(len(train_len))

                    def do_val_alt(the_loader, the_len):
                        the_loss = 0.0
                        for data in the_loader:
                            out = model(data.x, data.edge_index, data.batch, data.edge_mask)
                            the_loss += model.loss(out, data.y).sum()
                        # print('val mape: ', val_mape, len(val_set_alt))
                        the_loss = float(the_loss) / the_len
                        return the_loss

                    val_loss = do_val_alt(val_loader, int(len(self.dataset_loader.val_mask)))

                if val_loss < best_val_loss:  # New best results
                    if self.logging:
                        print("Val improved")
                        print(
                            f"Epoch: {epoch}, train_rmse: {train_loss:.6f}, "
                            f"val_loss: {val_loss:.6f}")

                    best_val_loss = val_loss
                    best_epoch = epoch
                    self.model_manager.store_checkpoint(self.model_name, self.dataset_name, model, train_loss, val_loss,
                                                        best_epoch)

                # Early stopping
                if epoch - best_epoch > self.early_stopping:
                    break
                pass
            else:
                assert 0

        if model.type == 'reg':
            self.model_manager.store_best_checkpoint(self.model_name, self.dataset_name, model, train_mape, val_mape,
                                                     best_epoch)
        elif model.type == 'cls':
            self.model_manager.store_best_checkpoint(self.model_name, self.dataset_name, model, train_loss, val_loss,
                                                     best_epoch)

        pass


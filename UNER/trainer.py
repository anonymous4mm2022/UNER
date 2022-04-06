import os
import logging
import shutil
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam, RMSprop

from data_loader import WebsiteProcessor, load_word_matrix
from utils import load_vocab, compute_metrics, report
from model import ACN,ACN1,ACN2,ACN3,ACN4,ACN5,ACN6,ACN7



logger = logging.getLogger(__name__)

OPTIMIZER_LIST = {
    "adam": Adam,
    "rmsprop": RMSprop
}


class OriTrainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset


        self.label_lst = WebsiteProcessor.get_labels()
        self.num_labels = len(self.label_lst)

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.word_vocab, self.char_vocab, self.word_ids_to_tokens, self.char_ids_to_tokens = load_vocab(args)
        self.pretrained_word_matrix = None
        if not args.no_w2v:
            self.pretrained_word_matrix = load_word_matrix(args, self.word_vocab)
        
    def clear_model(self):
        if os.path.exists(self.args.model_dir):
            shutil.rmtree(self.args.model_dir)
    def analysis_preds(self,out_label_list,preds_list):
        f = open(self.args.preds_file,'w',encoding='utf-8')
        for label1,label2 in zip(out_label_list,preds_list):
           f.write('{}|{}\n'.format(' '.join(label1),' '.join(label2)))
        f.close()


    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        # Save argument
        torch.save(self.args, os.path.join(self.args.model_dir, self.args.args_file))
        # Save model for inference
        #torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, 'model.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, self.args.model_file))
        #logger.info("Saving model checkpoint to {}".format(os.path.join(self.args.model_dir, 'model.pt')))
        logger.info("Saving model checkpoint to {}".format(os.path.join(self.args.model_dir, self.args.model_file)))

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            # self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            self.args = torch.load(os.path.join(self.args.model_dir, self.args.args_file))
            logger.info("***** Args loaded *****")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, self.args.model_file)))
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

class Trainer(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN(args, self.pretrained_word_matrix)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          #'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          #'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                #print(batch[5])
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])
            #print(len(out_label_list[i]),len(preds_list[i]))
        self.analysis_preds(out_label_list,preds_list)
        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results

class Trainer5(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer5, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN5(args, self.pretrained_word_matrix)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          #'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          #'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                #print(batch[5])
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        self.analysis_preds(out_label_list,preds_list)
        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results


class Trainer1(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer1, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN1(args, self.pretrained_word_matrix)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          #'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")
                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          #'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                #print(batch[5])
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])
        self.analysis_preds(out_label_list,preds_list)
        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results
class Trainer2(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer2, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN2(args, self.pretrained_word_matrix)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          #'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          #'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                #print(batch[5])
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        self.analysis_preds(out_label_list,preds_list)
        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results
class Trainer3(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer3, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN3(args, self.pretrained_word_matrix)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                #print(batch[5])
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])
        self.analysis_preds(out_label_list,preds_list)
        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results

class Trainer4(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer4, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN4(args, self.pretrained_word_matrix)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'img_feature': batch[2],
                          'dns_feature':batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5]}
                #print(batch[5])
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])



        self.analysis_preds(out_label_list,preds_list)

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results

class Trainer6(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer6, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN6(args)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                          'img_feature': batch[2],
                          'mask': batch[4],
                          'label_ids': batch[5],
                          'token_ids': batch[6],
                          'token_length': batch[7]
                         }
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                          'img_feature': batch[2],
                          'mask': batch[4],
                          'label_ids': batch[5],
                          'token_ids': batch[6],
                          'token_length': batch[7]
                         }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])



        self.analysis_preds(out_label_list,preds_list)

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results


class Trainer7(OriTrainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(Trainer7, self).__init__(args, train_dataset, dev_dataset, test_dataset)
                
        self.model = ACN7(args)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # optimizer and schedule (linear warmup and decay)
        if self.args.optimizer not in OPTIMIZER_LIST.keys():
            raise ValueError("Please choose the optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
        optimizer = OPTIMIZER_LIST[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size = %d", self.args.train_batch_size)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                          'img_feature': batch[2],
                          'dns_feature': batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5],
                          'token_ids': batch[6],
                          'token_length': batch[7],
                          'domain_token_ids': batch[8],
                          'domain_token_length': batch[9]
                         }
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.evaluate("dev")
                    self.evaluate("test")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                          'img_feature': batch[2],
                          'dns_feature': batch[3],
                          'mask': batch[4],
                          'label_ids': batch[5],
                          'token_ids': batch[6],
                          'token_length': batch[7],
                          'domain_token_ids': batch[8],
                          'domain_token_length': batch[9]
                         }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                # decode() in `torchcrf` returns list with best index directly
                preds = np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte()))
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, np.array(self.model.crf.decode(logits, mask=inputs['mask'].byte())), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])



        self.analysis_preds(out_label_list,preds_list)

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + report(out_label_list, preds_list))  # Get the report for each tag result

        return results
from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, beat_similarity, qb_norm
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config,
                 valid_data_loader,  train_data_loader=None, tokenizer=None, lr_scheduler=None, writer=None, qb_norm=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0
        self.use_beat = config.use_beat
        self.mode = config.mode
        self.qb_norm = qb_norm
        self.qbnorm_beta = config.qbnorm_beta
        self.qbnorm_k = config.qbnorm_k
        self.qbnorm_mode = config.qbnorm_mode
        self.metric = config.metric
        self.beta = 0.8
        self.video_dir = config.videos_dir
        if 'Dance' in self.video_dir:
            self.data_type = 'Music-Dance'
        else:
            self.data_type = 'Music-Motion'

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):

            data['music'] = data['music'].to(self.device)
            data['video'] = data['video'].to(self.device)
            data['music_beat'] = data['music_beat'].to(self.device)
            data['video_beat'] = data['video_beat'].to(self.device)

            text_embeds, video_embeds = self.model(data)
            output = sim_matrix_training(text_embeds['music_data'], video_embeds['video_data']) + \
                        self.beta * sim_matrix_training(video_embeds['video_beat_pool'], text_embeds['music_beat_pool'], 'xpool')

            loss = self.loss(output, self.model.clip_logit_scale)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip_logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        music_embed_arr = []
        vid_embed_arr = []

        if self.mode == 'double_loss':
            mus_beat_arr, vid_beat_arr = [], []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):

                data['video'] = data['video'].to(self.device)
                data['music'] = data['music'].to(self.device)
                data['music_beat'] = data['music_beat'].to(self.device)
                data['video_beat'] = data['video_beat'].to(self.device)

                music_embed, vid_embed = self.model(data)

                mus_beat_arr.append(music_embed['music_beat'].cpu())
                vid_beat_arr.append(vid_embed['video_beat_pool'].cpu())
                music_embed_arr.append(music_embed['music_data'])
                vid_embed_arr.append(vid_embed['video_data'])
                sims_batch = sim_matrix_training(music_embed['music_data'], vid_embed['video_data']) + \
                                sim_matrix_training(vid_embed['video_beat_pool'], music_embed['music_beat_pool'], 'xpool') * self.beta


                curr_loss = self.loss(sims_batch, self.model.clip_logit_scale)
                total_val_loss += curr_loss.item()

            music_embeds = torch.cat(music_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            mus_beats = torch.cat(mus_beat_arr)
            vid_beats = torch.cat(vid_beat_arr)

            self.model.beat_pool.cpu()
            mus_beats_pooled = self.model.beat_pool(vid_beats, mus_beats)
            self.model.beat_pool.cuda()

            mode = self.qbnorm_mode
            print(mode)

            if self.qb_norm is not None:
                if self.metric == 'v2t':
                    test_feature_sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds).cpu().detach()
                    test_beat_sims = sim_matrix_inference(vid_beats.unsqueeze(1), mus_beats_pooled.unsqueeze(2), 'xpool').cpu().detach()
                    test_sims =  test_feature_sims + self.beta * test_beat_sims

                    # vid_embed_arr = []
                    # vid_beat_arr = []
                    # for _, data in tqdm(enumerate(self.train_data_loader)):
                    #     data['video'] = data['video'].to(self.device)
                    #     data['music'] = data['music'].to(self.device)
                    #     data['music_beat'] = data['music_beat'].to(self.device)
                    #     data['video_beat'] = data['video_beat'].to(self.device)
                    #     _, vid_embed = self.model(data)
                    #     vid_embed_arr.append(vid_embed['video_data'])
                    #     vid_beat_arr.append(vid_embed['video_beat_pool'].cpu())      

                    # vid_embeds = torch.cat(vid_embed_arr)
                    # vid_beats = torch.cat(vid_beat_arr)

                    # self.model.beat_pool.cpu()
                    # mus_beats_pooled = self.model.beat_pool(vid_beats, mus_beats)
                    # self.model.beat_pool.cuda() 

                    # train_feature_sims = sim_matrix_inference(vid_embeds.unsqueeze(1), music_embeds).cpu().detach()
                    # train_beat_sims = sim_matrix_inference(vid_beats.unsqueeze(1), mus_beats_pooled.unsqueeze(2), 'xpool')

                    # We provide the similarity of the matrix here. The code will be completed after our thesis accepted
                    if self.data_type == 'Music-Dance':
                        train_feature_sims = torch.load('./checkout/Data2_v2m_train_feature_sims.pt')
                        train_beat_sims = torch.load('./checkout/Data2_v2m_train_beat_sims.pt')
                    else:
                        train_feature_sims = torch.load('./checkout/Data1_v2m_train_feature_sims.pt')
                        train_beat_sims = torch.load('./checkout/Data1_v2m_train_beat_sims.pt')                   


                    if mode == 'mode1':
                        train_sims = train_feature_sims + train_beat_sims * self.beta

                        train_sims = train_sims.cpu().detach()
                        train_sims, test_sims = np.squeeze(np.array(train_sims), axis=1), np.squeeze(np.array(test_sims), axis=1)
                        test_sims = np.transpose(test_sims, (1, 0))
                        test_sims = qb_norm(train_sims, test_sims, self.qbnorm_k, self.qbnorm_beta)
                        test_sims = np.transpose(test_sims, (1, 0))
                        test_sims = np.expand_dims(test_sims, axis=1)
                        test_sims = torch.from_numpy(test_sims)
                    else:
                        print('it is mode 2')
                        train_feature_sims = train_feature_sims.cpu().detach()
                        train_feature_sims, test_feature_sims = np.squeeze(np.array(train_feature_sims), axis=1), np.squeeze(np.array(test_feature_sims), axis=1)
                        test_feature_sims = np.transpose(test_feature_sims, (1, 0))
                        test_feature_sims = qb_norm(train_feature_sims, test_feature_sims, self.qbnorm_k, self.qbnorm_beta)
                        test_feature_sims = np.transpose(test_feature_sims, (1, 0))
                        test_feature_sims = np.expand_dims(test_feature_sims, axis=1)
                        test_feature_sims = torch.from_numpy(test_feature_sims)

                        train_beat_sims = train_beat_sims.cpu().detach()
                        train_beat_sims, test_beat_sims = np.squeeze(np.array(train_beat_sims), axis=1), np.squeeze(np.array(test_beat_sims), axis=1)
                        test_beat_sims = np.transpose(test_beat_sims, (1, 0))
                        test_beat_sims = qb_norm(train_beat_sims, test_beat_sims, self.qbnorm_k, self.qbnorm_beta)
                        test_beat_sims = np.transpose(test_beat_sims, (1, 0))
                        test_beat_sims = np.expand_dims(test_beat_sims, axis=1)
                        test_beat_sims = torch.from_numpy(test_beat_sims)

                        test_sims = test_feature_sims + self.beta * test_beat_sims

                else:
                    test_feature_sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds).cpu().detach()
                    test_beat_sims = sim_matrix_inference(vid_beats.unsqueeze(1), mus_beats_pooled.unsqueeze(2), 'xpool').cpu().detach()
                    test_sims =  test_feature_sims + self.beta * test_beat_sims

                    # music_embed_arr = []
                    # mus_beat_arr = []
                    # for _, data in tqdm(enumerate(self.train_data_loader)):
                    #     data['video'] = data['video'].to(self.device)
                    #     data['music'] = data['music'].to(self.device)
                    #     data['music_beat'] = data['music_beat'].to(self.device)
                    #     data['video_beat'] = data['video_beat'].to(self.device)
                    #     music_embed, _ = self.model(data)
                    #     music_embed_arr.append(music_embed['music_data'])
                    #     mus_beat_arr.append(music_embed['music_beat'].cpu())

                    # music_embeds = torch.cat(music_embed_arr)
                    # mus_beats = torch.cat(mus_beat_arr)

                    # self.model.beat_pool.cpu()
                    # mus_beats_pooled = self.model.beat_pool(vid_beats, mus_beats)
                    # self.model.beat_pool.cuda()

                    # train_feature_sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds).cpu()
                    # train_beat_sims = torch.transpose(sim_matrix_inference(vid_beats.unsqueeze(1), mus_beats_pooled.unsqueeze(2), 'xpool'), 0, 2)

                    # We provide the similarity of the matrix here. The code will be completed after our thesis accepted
                    if self.data_type == 'Music-Dance':
                        train_feature_sims = torch.load('./checkout/Data2_m2v_train_feature_sims.pt')
                        train_beat_sims = torch.load('./checkout/Data2_m2v_train_beat_sims.pt')
                    else:
                        train_feature_sims = torch.load('./checkout/Data1_m2v_train_feature_sims.pt')
                        train_beat_sims = torch.load('./checkout/Data1_m2v_train_beat_sims.pt')    

                    if mode == 'mode1':
                        train_sims = train_feature_sims + self.beta * train_beat_sims

                        train_sims = train_sims.cpu().detach()
                        train_sims, test_sims = np.squeeze(np.array(train_sims), axis=1), np.squeeze(np.array(test_sims), axis=1)
                        test_sims = qb_norm(train_sims, test_sims, self.qbnorm_k, self.qbnorm_beta)
                        test_sims = np.expand_dims(test_sims, axis=1)
                        test_sims = torch.from_numpy(test_sims)
                    else:
                        print('it is mode 2')
                        train_feature_sims = train_feature_sims.cpu().detach()
                        train_feature_sims, test_feature_sims = np.squeeze(np.array(train_feature_sims), axis=1), np.squeeze(np.array(test_feature_sims), axis=1)
                        test_feature_sims = qb_norm(train_feature_sims, test_feature_sims, self.qbnorm_k, self.qbnorm_beta)
                        test_feature_sims = np.expand_dims(test_feature_sims, axis=1)
                        test_feature_sims = torch.from_numpy(test_feature_sims)

                        train_beat_sims = train_beat_sims.cpu().detach()
                        train_beat_sims, test_beat_sims = np.squeeze(np.array(train_beat_sims), axis=1), np.squeeze(np.array(test_beat_sims), axis=1)
                        test_beat_sims = qb_norm(train_beat_sims, test_beat_sims, self.qbnorm_k, self.qbnorm_beta)
                        test_beat_sims = np.expand_dims(test_beat_sims, axis=1)
                        test_beat_sims = torch.from_numpy(test_beat_sims)

                        test_sims = test_feature_sims + self.beta * test_beat_sims
                        

            else:
                print(vid_beats.unsqueeze(1).shape, mus_beats.unsqueeze(2).shape)
                test_sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds).cpu().detach() + \
                            self.beta * sim_matrix_inference(vid_beats.unsqueeze(1), mus_beats_pooled.unsqueeze(2), 'xpool')

            sims = test_sims
            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res

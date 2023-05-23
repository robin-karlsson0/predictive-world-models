import cv2
import numpy as np
import torch

from vdvae.inference import vdvaeInferenceModule


class WorldModel(vdvaeInferenceModule):
    '''
    '''

    def __init__(self):

        super().__init__()

    def inference(self, x: torch.Tensor, temp: float = 1.) -> torch.Tensor:
        '''
            x: (B,C,H,W)
        '''
        x = x.cuda()
        x_hat = self.forward(x, temp)
        x_hat = x_hat.cpu()
        return x_hat

    def sample(self,
               x: torch.Tensor,
               bs: int,
               temp: float = 1.) -> torch.Tensor:
        '''
        Args:
            x: (C,H,W)
            bs: Batch size (i.e. simultaneous sampling)

        Returns:
            List of 'x_hat' (C,H,W) tensors.
        '''
        x = x.unsqueeze(0)
        x = torch.tile(x, (bs, 1, 1, 1))
        x_hat = self.inference(x, temp)
        x_hat = x_hat.chunk(bs)
        # Remove batch index
        x_hat = [x[0] for x in x_hat]

        return x_hat

    def sat_sample(self,
                   x: torch.Tensor,
                   traj: torch.Tensor,
                   bs: int,
                   temp: float = 1.,
                   max_tries=10) -> torch.Tensor:
        '''
        Returns a predicton tensor 'x' satisfying constraints.

        NOTE Returns random sample if run out of tries to satisfy constraints.

        Args:
            x: Input tensor (C,H,W)
            traj: (H,W)
            bs: Batch size (i.e. simultaneous sampling)
            max_tries: Maximum number of constrained samples before returning
                       random sample.

        Returns:
            x: Prediction tensor (C,H,W) satisfying constraints.
        '''
        # Number of elements representing complete trajectory
        traj_elem_full = torch.sum(traj)

        # Condition input
        # mask = traj == 1
        # x[0, mask] = 1.

        traj = traj.unsqueeze(0)
        traj = torch.tile(traj, (bs, 1, 1))

        sample_idx = 0
        while sample_idx < max_tries:

            # Convert list of 'x_hat's to (B,C,H,W) tensor
            x_hat = self.sample(x, bs, temp)
            x_hat = torch.stack(x_hat)

            # Element-wise multiplication ==> Only predicted traj elements 1
            traj_elem = x_hat[:, 0] * traj
            traj_elem = torch.sum(traj_elem, dim=(1, 2))

            sat_samples = traj_elem == traj_elem_full
            num_sat_samples = torch.sum(sat_samples).item()

            if num_sat_samples > 0:
                # Get idx of first element that is true
                batch_idx = next(idx for idx, val in enumerate(sat_samples)
                                 if val == True)
                x_hat = x_hat[batch_idx]
                return x_hat

            sample_idx += 1

        # Give up and return a random sample
        x_hat = self.sample(x, bs, temp)
        return x_hat[0]


#        tot_sat_samples = 0
#        sample_tries = 0
#        x_hats = []
#
#        while tot_sat_samples < n and sample_tries < max_sample_tries:
#            x_hat = self.sample(x, batch_size, temp=temp)
#            x_hat = torch.stack(x_hat)
#
#            # Element-wise multiplication ==> Only predicted traj elements 1
#            traj_elem = x_hat[:, 0] * traj
#            traj_elem = torch.sum(traj_elem, dim=(1, 2))
#
#            sat_samples = traj_elem == traj_elem_full
#            num_sat_samples = torch.sum(sat_samples).item()
#
#            if num_sat_samples > 0:
#                x_hat = x_hat[sat_samples]
#                x_hats.append(x_hat)
#                tot_sat_samples += num_sat_samples
#
#            sample_tries += 1
#
#        if len(x_hats) > 0:
#            x_hats = torch.concat(x_hats)
#            x_hats = x_hats[:n]
#            return x_hats
#        else:
#            x_hats = self.sample(x, n, temp=temp)
#            return x_hats[0]

    def sample_best(
        self,
        x: torch.Tensor,
        n: int,
        bs: int,
        x_pnt: int,
        y_pnt: int,
        temp: float = 1.,
    ) -> torch.Tensor:
        '''
        Args:
            x_hat: (C,H,W)
            n: Number of samples to compare for goodness.
            bs: Batch size (i.e. simultaneous sampling).
            x_pnt: Coordinates of ego vehicle for determining road.
            y_pnt:
            temp:
        '''
        x_hats = []
        # Do enough sample iterations to exceed 'n' total samples
        for _ in range(n // bs + 1):
            x_hat = self.sample(x, bs, temp)
            x_hats += x_hat

        scores = []
        for x_hat in x_hats:
            score = self.cal_score(x_hat, x_pnt, y_pnt)
            scores.append(score)

        best_idx = np.argmax(scores)

        return x_hats[best_idx]

    def cal_score(self, x_hat: torch.Tensor, x_pnt: int, y_pnt: int):
        '''
        Args:
            x_hat: (C,H,W)
            x_pnt: Coordinates of ego vehicle for determining road.
            y_pnt:
        '''
        road = x_hat[0].cpu().numpy().copy().astype(np.uint8)

        # Ignore road elements not connected to ego vehicle road
        road_connected = np.zeros_like(road, dtype=np.uint8)
        contours, _ = cv2.findContours(255 * road, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            is_inside = cv2.pointPolygonTest(contour, (x_pnt, y_pnt), False)
            if is_inside == 1:
                cv2.drawContours(road_connected, [contour], -1, 255, -1)
                break
        road_connected = road_connected / 255

        # Number of road boundary elements
        try:
            road_connected[1:-1, 1:-1] = 0
            num_boundary = np.sum(road_connected)
        # Handle unknown error causing road_connected to no be a matrix
        except:
            num_boundary = 0

        return num_boundary

    def sat_sample_best(
        self,
        x: torch.Tensor,
        traj: torch.Tensor,
        num_sampling: int,
        x_pnt: int,
        y_pnt: int,
        batch_size: int = 1,
        temp: float = 1.,
    ) -> torch.Tensor:
        '''
        Collect N samples satisfying future trajectory constraints and return
        best sample among them.

        NOTE Returns random sample if run out of tries to satisfy constraints.

        Args:
            x: Input tensor (C,H,W)
            num_sampling: Number of samples to compare for goodness.
            x_pnt: Coordinates of ego vehicle for determining road.
            y_pnt:
            temp:

        Returns:
            x: Prediction tensor (C,H,W) satisfying constraints.
        '''
        x_hats = []
        for _ in range(num_sampling):
            x_hat = self.sat_sample(x, traj, batch_size, temp=temp)
            x_hats.append(x_hat)

        scores = []
        for x_hat in x_hats:
            score = self.cal_score(x_hat, x_pnt, y_pnt)
            scores.append(score)

        best_idx = np.argmax(scores)
        x_best = x_hats[best_idx]
        return x_best

import numpy as np
import torch
import torch.nn as nn

class DiceScoreCoefficient(nn.Module):
    def __init__(self, n_classes):
        super(DiceScoreCoefficient, self).__init__()
        self.n_classes = n_classes
        # confusion matrix는 이제 forward 안에서 새로 계산할 필요 없음
        # 대신 배치 전체에 대한 누적 히스토그램을 forward 마지막에 사용

    # fast_hist 함수는 사용하지 않는 것 같아서 일단 놔둘게 (아직 필요 없으니)
    def fast_hist(self, label_true, label_pred, labels):
        mask = (label_true >= 0) & (label_true < labels)
        hist = np.bincount(
            labels * label_true[mask].astype(int) + label_pred[mask],
            minlength=labels ** 2
        ).reshape(labels, labels)
        return hist

    # Dice Score (F1) 계산 함수 (이건 아까 수정해줬던 대로)
    def _dsc(self, mat):
        tp = np.diag(mat) # True Positive
        fp = mat.sum(axis=1) - tp # False Positive
        fn = mat.sum(axis=0) - tp # False Negative

        # Precision과 Recall 계산 (0으로 나누는 경우 방지)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float32), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float32), where=(tp + fn) != 0)

        # Dice Score (F1 Score) 계산 (0으로 나누는 경우 방지)
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        dice_score = np.divide(2. * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=np.float32), where=(precision + recall) != 0)

        return dice_score # Dice Score 반환

    ### main ###
    # 배치 전체를 한 번에 처리하도록 forward 함수 대폭 수정!
    def forward(self, output, target, argmax=True):
        # Softmax 후 argmax로 클래스 채널 제거 -> 예측값 (클래스 인덱스) [B, H, W]
        if argmax:
            output = output.float()
            output = torch.softmax(output, dim=1)
            seg = torch.argmax(output, dim=1)  # [B, H, W]
        else:
            # argmax=False로 호출하는 경우는 드물겠지만, 이때는 output이 이미 [B, H, W] 형태라고 가정
            seg = output

        # 타겟을 원-핫 인코딩 [B, C, H, W] 에서 클래스 인덱스 [B, H, W] 로 변환
        # CustomDataset 결과 print를 봤을 때 target은 [1, C, H, W] 형태이므로 argmax(dim=1) 적용
        if target.dim() == 4 and target.shape[1] == self.n_classes:
            target_indices = torch.argmax(target, dim=1) # [B, H, W]
        elif target.dim() == 3 and target.shape[1:] == seg.shape[1:]:
             # 혹시 target이 이미 [B, H, W] 형태일 경우 (rare but safe check)
             target_indices = target
        else:
             # 예상치 못한 target 형태면 에러 발생시키기
             raise ValueError(f"Unexpected target shape: {target.shape}. Expected [B, C, H, W] or [B, H, W] (where C is num_classes).")

        # Convert tensors to numpy arrays
        seg_np = seg.cpu().numpy() # [B, H, W]
        target_indices_np = target_indices.cpu().numpy() # [B, H, W]

        # !!! 에러 해결 핵심 부분 !!!
        # 배치와 공간 차원 (H, W)을 모두 합쳐서 1차원으로 쭉 펴기 (flatten)
        seg_flat = seg_np.flatten() # [B*H*W]
        target_flat = target_indices_np.flatten() # [B*H*W]

        # 유효한 값 마스크 생성 (0 미만이거나 n_classes 이상인 값 제외)
        mask_flat = (target_flat >= 0) & (target_flat < self.n_classes)

        # 마스크를 적용하여 유효한 예측값과 정답값만 추출 (결과는 1차원 배열)
        target_valid = target_flat[mask_flat].astype(int)
        seg_valid = seg_flat[mask_flat].astype(int)

        # np.bincount를 사용하여 배치 전체의 유효한 값들로 Confusion Matrix 계산
        # np.bincount는 1차원 배열을 받으므로 flatten된 데이터에 바로 적용 가능
        # self.n_classes * target_valid + seg_valid 계산으로 각 (정답, 예측) 쌍을 고유한 인덱스로 만듦
        # minlength는 n_classes * n_classes 로 설정하여 모든 가능한 쌍의 빈도를 계산
        hist = np.bincount(
            self.n_classes * target_valid + seg_valid,
            minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes) # 결과를 n_classes x n_classes 형태로 reshape

        # 계산된 confusion matrix로 Dice Score 계산 후 반환
        dsc = self._dsc(hist)

        # _dsc는 클래스별 스코어 배열 [n_classes]를 반환함
        return dsc

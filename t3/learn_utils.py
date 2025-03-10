

import torch

# def compute_alignment_loss(student_features, teacher_features, mask):
#     # get the masked features
#     masked_student_features = student_features[mask]
#     masked_teacher_features = teacher_features[mask]
    
#     # L1 LOSS
#     alignment_loss = torch.mean(torch.abs(masked_student_features - masked_teacher_features))
    
#     return alignment_loss

# def compute_reconstruction_loss(predicted, ground_truth, mask):
#     # get the masked values
#     masked_predicted = predicted[mask]
#     masked_ground_truth = ground_truth[mask]
    
#     # l2 loss
#     reconstruction_loss = torch.mean((masked_predicted - masked_ground_truth) ** 2)
    
#     return reconstruction_loss


# def compute_total_loss(student_features, teacher_features, predicted, ground_truth, mask, lambda_align=1.0):
#     align_loss = compute_alignment_loss(student_features, teacher_features, mask)
#     rec_loss = compute_reconstruction_loss(predicted, ground_truth, mask)

#     total_loss = align_loss + lambda_align * rec_loss
#     return total_loss



def compute_encoder_distance(f_teacher, f_student, reduction="mean"):
    """
    get the distance between the teacher and student encoders
    
    Args:
        f_teacher (Tensor): 教师编码器的输出，形状为 (batch_size, L, D)
        f_student (Tensor): 学生编码器的输出，形状为 (batch_size, L, D)
        reduction (str): 'mean' 表示取平均距离
        
    Returns:
        distance (Tensor): 计算得到的距离值
    """
    # L2 distance between teacher and student features
    diff = f_teacher - f_student  # 形状仍为 (batch_size, L, D)
    # 在特征维度上计算 L2 距离
    per_sample_distance = torch.norm(diff, p=2, dim=-1)  # (batch_size, L)
    
    # 可以对所有位置求平均或先对每个样本求平均，再对所有样本求平均
    if reduction == "mean":
        return per_sample_distance.mean()
    else:
        return per_sample_distance

# a.Viết công thức toán, implement (code numpy) và giải thích về Triplet loss.
'''
    Định nghĩa Triplet loss:
    L(a,p,n)=max(0,∥f(a)−f(p)∥^2 −∥f(a)−f(n)∥^2 + α)
    f : biến các giá trị thành các vector đặc trưng
    a : mẫu chính 
    p : mẫu dương so với a
    n : mẫu âm so với a
    α : margin 
    Triplet Loss được thiết kế để đảm bảo rằng khoảng cách giữa mẫu chính 
    và mẫu dương nhỏ hơn khoảng cách giữa mẫu chính và mẫu âm ít nhất là 𝛼
    Điều này giúp mô hình học cách đưa các mẫu cùng lớp gần nhau và các mẫu
    khác lớp xa nhau trong không gian đặc trưng.
    '''
import numpy as np
def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.sum((anchor - positive) ** 2)
    neg_dist = np.sum((anchor - negative) ** 2)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return loss

anchor = np.array([1, 2])
positive = np.array([1, 2.1])
negative = np.array([3, 4])
loss = triplet_loss(anchor, positive, negative)
print(f'Triplet Loss: {loss}')

#b. Viết công thức toán, implement và giải thích khi Input triplet loss mở rộng không chỉ là 1 mẫu thật và một mẫu giả nữa mà sẽ là 2 mẫu thật và 5 mẫu giả
'''L(a,P,N)=∑p∈P ∑n∈N max(0,∥f(a)−f(p)∥^2 − ∥f(a)−f(n)∥^2 +α)
    a : mẫu chính
    P : mẫu thật
    N : mẫu giả 
    α : margin
    Triplet loss là hàm mất mát đảm bảo rằng khoảng cách giữa mẫu 
    chính và mẫu dương sẽ nhỏ khoảng cách giữa mẫu chính và mẫu âm,
    nghĩa triplet sẽ càng nhỏ khi mẫu chính gần mẫu dương và xa mẫu âm.'''
def triplet_loss_extended(anchor, positives, negatives, alpha=0.2):
    total_loss = 0.0
    for positive in positives:
        for negative in negatives:
            pos_dist = np.sum((anchor - positive) ** 2)
            neg_dist = np.sum((anchor - negative) ** 2)
            loss = np.maximum(0, pos_dist - neg_dist + alpha)
            total_loss += loss
    return total_loss

# Example usage
anchor = np.array([1, 2])
positives = [np.array([1, 1.5]), np.array([1.1, 2.2])]
negatives = [np.array([3, 4]), np.array([5, 6]), np.array([6, 8]), np.array([5.5, 7.7]), np.array([3.4, 4.4])]
loss = triplet_loss_extended(anchor, positives, negatives)
print(f'Extended Triplet Loss: {loss}')

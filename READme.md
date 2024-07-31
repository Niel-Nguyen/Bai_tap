# Bài 1: 
## Xây dựng một mô hình Machine learning (not deep learning) ứng dụng cho bài phân biệt loại ký tự quang học, ứng dụng data MNIST. Chỉ sử dụng numpy.
### Down load dataset của Mnist [tại đây](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data)

Dùng thuật toán KNN cho bài toán phân loại với data Mnist.

|Data|Image|Label|
|----|----|----|
|Train|50000|50000|
|Test|10000|10000|

Test ảnh bên ngoài:
![example](image.png)
**predict label is: 5**
***

# Bài 2: Triplet loss
##  a. Viết công thức toán, implement (code numpy) và giải thích về Triplet loss.
 ##

### Triplet Loss được sử dụng để học một không gian nhúng (embedding space) sao cho khoảng cách giữa một điểm anchor và một điểm positive (cùng lớp) nhỏ hơn khoảng cách giữa điểm anchor và một điểm negative (khác lớp) với một biên độ margin nhất định.

**Công thức Triplet Loss**

**\[ L(a, p, n) = \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha) \]**

Trong đó:

- \( L(a, p, n) \): Giá trị Triplet Loss.
- \( a \): Vector đặc trưng của điểm anchor.
- \( p \): Vector đặc trưng của điểm positive.
- \( n \): Vector đặc trưng của điểm negative.
- \( \|f(a) - f(p)\|^2 \): Khoảng cách Euclidean bình phương giữa vector đặc trưng của anchor và positive.
- \( \|f(a) - f(n)\|^2 \): Khoảng cách Euclidean bình phương giữa vector đặc trưng của anchor và negative.
- \( \alpha \): Margin (biên độ) để đảm bảo rằng khoảng cách giữa anchor và positive phải nhỏ hơn khoảng cách giữa anchor và negative một lượng ít nhất là \(\alpha\).

Công thức này đảm bảo rằng các mẫu cùng lớp sẽ gần nhau hơn trong không gian nhúng, trong khi các mẫu khác lớp sẽ cách xa nhau một khoảng cách tối thiểu.

## b. Viết công thức toán, implement (code numpy) và giải thích khi Input triplet loss mở rộng không chỉ là 1 mẫu thật và một mẫu giả nữa mà sẽ là 2 mẫu thật và 5 mẫu giả.
# Triplet Loss Mở Rộng

Triplet Loss được sử dụng để học một không gian nhúng (embedding space) sao cho khoảng cách giữa một điểm anchor và một điểm positive (cùng lớp) nhỏ hơn khoảng cách giữa điểm anchor và một điểm negative (khác lớp) với một biên độ margin nhất định. Trong trường hợp mở rộng, chúng ta có nhiều hơn một mẫu positive và nhiều hơn một mẫu negative.

**Công thức Triplet Loss Mở Rộng**

**\[ L(a, P, N) = \sum_{p \in P} \sum_{n \in N} \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha) \]**

Trong đó:

- \( L(a, P, N) \): Giá trị Triplet Loss.
- \( a \): Vector đặc trưng của điểm anchor.
- \( P \): Tập hợp các vector đặc trưng của các điểm positive.
- \( N \): Tập hợp các vector đặc trưng của các điểm negative.
- \( \|f(a) - f(p)\|^2 \): Khoảng cách Euclidean bình phương giữa vector đặc trưng của anchor và một positive bất kỳ trong tập \( P \).
- \( \|f(a) - f(n)\|^2 \): Khoảng cách Euclidean bình phương giữa vector đặc trưng của anchor và một negative bất kỳ trong tập \( N \).
- \( \alpha \): Margin (biên độ) để đảm bảo rằng khoảng cách giữa anchor và positive phải nhỏ hơn khoảng cách giữa anchor và negative một lượng ít nhất là \(\alpha\).

Công thức này đảm bảo rằng các mẫu cùng lớp sẽ gần nhau hơn trong không gian nhúng, trong khi các mẫu khác lớp sẽ cách xa nhau một khoảng cách tối thiểu.

***
#   Bài 3:

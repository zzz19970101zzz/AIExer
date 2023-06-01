from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:] + s[:n]
    def reverseWords(self, s: str) -> str:
        res = ''
        s.rstrip()
        s.lstrip()
        l = s.split(' ')
        for str1 in l[::-1]:
            if str1 == '':
                continue
            res += str1
            if str1 != l[0]:
                res += ' '
        return res

if __name__ == '__main__':
    s = Solution;
    print(s.reverseLeftWords(s,'abcdeffg', 2))
    print(s.reverseWords(s,'a good  example'))
    m = 100
    x = 6 * np.random.rand(m,1) - 3
    y = 0.5 * x**2 + x + 2 + np.random.randn(m,1)
    poly_features = PolynomialFeatures(degree=3,include_bias=False)
    x_ploy = poly_features.fit_transform(x)
    print(x[0])
    print(x_ploy[0])



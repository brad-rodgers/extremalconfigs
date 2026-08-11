# extremalconfigs
Programs for finding extremal configurations in the weighted Hilbert inequality. This has been used to make conjectures below. Code has been produced by ChatGPT 5.6 Sol and has not been independently audited at this time.

# Hilbert inequality and reformulation
For fixed $N \geq 2$ and fixed $\lambda_1 < \cdots < \lambda_N$, let $C_N(\lambda_1,...,\lambda_N)$ be the optimal constant $C$ in the inequality

$$\Big| \sum_{\substack{1 \leq m,n \leq N \\ m \neq n}}  \frac{w_m \overline{w_n}}{\lambda_m - \lambda_n} \Big| \leq C \sum_{1\leq m \leq N} \frac{|w_m|^2}{\delta_m},$$

where $\delta_n = \min_{m\neq n}|\lambda_m-\lambda_n|$ and $w_1,...,w_N$ are arbitrary real or complex numbers. This may be phrased in terms of the spectral radius of the $N\times N$ skew-symmetric matrix $B_N(\lambda_1,...,\lambda_N)$ with entries $b_{jk}$ given by

$$b_{jk} = \begin{cases} 
\sqrt{\delta_j \delta_k}/(\lambda_j - \lambda_k) & \textrm{for}\; j\neq k \\
0 & \textrm{for}\; j = k.
\end{cases}$$

Let

$$C_N = \sup_\lambda \rho(B_N(\lambda_1,...,\lambda_N)),$$

where $\rho$ is the spectral radius. It is an exercise in analysis using translation and dilation invariance in $\lambda$ of $B_N$ to see that the supremum is actually realized for each $N$. Because of this invariance, we lose no generality in looking for extremal $\lambda$ with $\lambda_1 = 1$ and such that the smallest gap between any of the $\lambda_j$ is $1$. We can thus describe a configuration $\lambda_1,...,\lambda_N$ by listing the $N-1$ gaps which we label $g_1,...,g_{N-1}$. It will be convenient in discussing gap lists to introduce the notation that if a gap size $g$ is repeated $r$ times, we write this as $g^{[r]}$. So for instance the configuration $\lambda_1 < \cdots < \lambda_5$ given by $0 < 1 < 4 < 7 < 7.5$ would be described by the gap list $1, 3^{[2]}, 0.5$.

The numerical data below allows one to somewhat tenuously make the following conjectures about gap lists $g_1,...,g_{N-1}$ for extremal configurations in dimension $N$:

(1) The gap list is symmetric; that is $g_{N-j} = g_j$ for all $N$ and $j$.

(2)  For each $N$ the gap list can be written

$$\gamma_L^{[n_L]},...,\gamma_1^{[n_1]}, \gamma_0^{[n_0]}, \gamma_1^{[n_1]},..., \gamma_L^{[n_L]},$$

for some $L = L_N$ and values $\gamma_0,...,\gamma_L$ and $n_0,...,n_L$ also depending on $N$ with the normalization $\gamma_0 = 1$, and with the further properties that $\gamma_{j+1} > \gamma_j$ while $n_{j+1} \leq n_j$ for all $j$.

(3) $L_N = \log_\pi(N) + O(1)$.

(4) $n_0 \sim \left(1-\frac{1}{\pi}\right)N$ as $N\rightarrow\infty$.

(5) $n_j \sim \frac{1}{2}\left(1-\frac{1}{\pi}\right) \frac{1}{\pi^j} N$ as $N\rightarrow\infty$ for all fixed $j\geq 1$.
\end{enumerate}

Furthermore for each fixed $j$ it seems $\gamma_j$ grows polynomially fast with $N$, at a rate which increases the larger $j$ is. It may even be that $\gamma_j = N^{j+o(1)}$, but the evidence for this is not yet especially strong.

# ExtremalLambdaSearch_SmallN

This directory contains a numerical search for extremal $\lambda$ for $2 \leq N \leq 48$. Some representative examples include the following: For $2 \leq N \leq 20$ it appears numerically that extremal configurations are those with the gap list,

$$1^{[N-1]}.$$

For $N = 21$ this pattern of equal spacings breaks down. Numerically the extremal gap list appears to be approximately

$$a, 1^{[18]}, a \quad \textrm{where}\; a = 3.47$$

For $N = 46$ the extremal gap list appears to be approximately

$$a, b^{[2]}, 1^{[39]}, b^{[2]}, a \quad \textrm{where}\; a = 19.15, \, b = 5.71$$

These candidates for extremal gap lists were found via a multistart optimization routine in a high dimensional nonconvex domain; so while it is reasonable to believe they are close approximations to the actually extremal configurations, no proof of this exists for general $N$.

# ExtremalLambdaSearch_Experiments

This directory contains numerical experiments related to the large $N$ conjectures given above for gap distributions.

A numerical analysis based on extrapolating these patterns suggest that the optimal absolute constant $C$ in \eqref{eq:infinite_Hilbert} is around 3.1432.

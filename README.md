# Parallel Multi-Start Optimization

### Information
1) **Cost Function**: We used custom cost function to show effectiveness of the model, but you can apply it to any cost function as you wish:

$$ f(x, y) = -\left( \sum_{i=1}^{n} w_i \exp\left(-\left(\frac{(x - p_{i,x})^2}{2\sigma_{x,i}^2} + \frac{(y - p_{i,y})^2}{2\sigma_{y,i}^2}\right)\right) + 0.2 \sin(4\pi x) \cos(3\pi y) - 0.1 \left(\frac{1}{5}x^2 + \frac{4}{5}y^2\right) \right) $$

$$ \begin{align*}
n & : \text{Number of minima} \\
x, y & : \text{Input variables} \\
w_i & : \text{Weight associated with each Gaussian well} \\
p_{i,x}, p_{i,y} & : \text{Position of the $i$-th Gaussian well in the $x$ and $y$ dimensions} \\
\sigma_{x,i}, \sigma_{y,i} & : \text{Standard deviations for the $x$ and $y$ dimensions of the $i$-th Gaussian well}
\end{align*} $$

2) **Optimization Method**: We used Gradient Descent for optimizing the cost function
    - Of course there is better approaches exist, but our purpose is show the effectiveness of multi-start optimization

$$ \theta_{k+1} = \theta_k - \mu \cdot \nabla J(\theta) $$

$$ \begin{align*}
\theta_k & : \text{Current position} \\
\theta_{k+1} & : \text{Next position} \\
\mu & : \text{Learning rate} \\
\nabla J(\theta)& : \text{Gradient of the cost function} \\
\end{align*} $$

- If we take the gradient of the cost function w.r.t _x_ and _y_

$$ \frac{\partial f}{\partial x} = -\left( \sum_{i=1}^{n} w_i \exp\left(-\left(\frac{(x - p_{i,x})^2}{2\sigma_{x,i}^2} + \frac{(y - p_{i,y})^2}{2\sigma_{y,i}^2}\right)\right) \cdot \frac{(x - p_{i,x})}{\sigma_{x,i}^2} + 0.2 \cdot 4\pi \cos(4\pi x) \cos(3\pi y) - 0.2x \right)$$

$$ \frac{\partial f}{\partial y} = -\left(\sum_{i=1}^{n} w_i \exp\left(-\left(\frac{(x - p_{i,x})^2}{2\sigma_{x,i}^2} + \frac{(y - p_{i,y})^2}{2\sigma_{y,i}^2}\right)\right) \cdot \frac{(y - p_{i,y})}{\sigma_{y,i}^2} - 0.2 \cdot 3\pi \sin(4\pi x) \sin(3\pi y) - 0.8y \right) $$

3) **Files**: There are 6 files:
    - _parameters.py_: Contains the parameters of the cost function. You can adjust it as you wish
    - _main.py_: Contains the cost function, optimization method and gradient of the cost function. You can adjust it as you wish
    - _single.py_: This is traditional single-start optimization. You can observe that it will probably stuck on local minima
    - _sequential.py_: This uses multi-start technique implemented sequentially
    - _parallel-mpi.py_: This uses parallel multi-start technique implemented with MPI library
    - _parallel-numba.py_: This uses parallel multi-start technique implemented with Numba's OMP

### Prerequistes
1. Make sure you have Python installed (preferably version 3.7 or later).
2. Install the required dependencies using `pip3`:
   ```bash
   pip3 install -r requirements.txt

You can execute these files using below commands:

    ```bash
    python3 main.py

    ```bash
    python3 single.py

    ```bash
    python3 sequential.py <num of starting points>

    ```bash
    mpiexec -n <num of processes> python3 parallel_mpi.py <num of starting points>

    ```bash
    python3 parallel_numba.py <num of starting points> <num of threads>




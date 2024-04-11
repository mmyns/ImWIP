import numpy as np
from tqdm import tqdm  # Assuming you're using tqdm for the progress bar

from tqdm import tqdm




def _line_search(f, fx, grad, x, d, t=0.5, c=1e-4, a=1.0, max_iter=20, verbose=False):
        m = np.dot(grad, d/np.linalg.norm(d))
        iterations = 0
        new_f = f(x + a*d)
        while iterations < max_iter and new_f > fx + a*c*m:
            if verbose:
                print(iterations, new_f, fx + a*c*m)
            a *= t
            new_f = f(x + a*d)
            iterations += 1
        return a


class SplitBarzilaiBorwein:
    def __init__(
        self,
        grad_f,
        x0,
        f=None,
        line_search_iter=None,
        init_step=None,
        bounds=None,
        projector=None,
        callback=None,
        norm=True,
    ):
        self.grad_f = grad_f
        self.x = [np.array(var) for var in x0]
        self.f = f
        self.line_search_iter = line_search_iter
        self.init_step = init_step
        self.bounds = bounds
        self.projector = projector
        self.callback = callback
        self.norm = norm
        self.iter = 0

        # Placeholder for variables that need to be updated
        self.stepsize = self.init_step
        self.gradp = None
        self.xp = None

    

    def _get_split_stepsize(self,x, grad, f=None, init_step=None, line_search_iter=None, verbose=False, norm = True):
        if init_step is not None:
            if norm:
                a = [0]*len(x)
                for i in range(len(x)):
                    n_grad = np.linalg.norm(grad[i])
                    print(n_grad)
                    print(init_step[i])
                    if n_grad > 0:
                        a[i] = init_step[i]/np.linalg.norm(grad[i])
            else:
                a = init_step
        else:
            if f is None:
                if line_search_iter is not None:
                    raise ValueError("f should be given for line_search")
                a = [1/np.linalg.norm(np.concatenate(grad))]*len(x)
            else:
                if line_search_iter is None:
                    line_search_iter = 20
                a = [0]*len(x)
                for i in range(len(x)):
                    n_grad = np.linalg.norm(grad[i])
                    if n_grad > 0:
                        def f_partial(xi, i):
                            x_edit = x[:]
                            x_edit[i] = xi
                            return f(*x_edit)
                        a = _line_search(
                            lambda xi: f_partial(xi, i),
                            f_partial(x[i], i),
                            grad[i],
                            x[i],
                            grad[i],
                            a=1/n_grad,
                            max_iter=line_search_iter,
                            verbose=True
                            )
        return a

    def run(self):
        grad = self.grad_f(*self.x)
        
        if self.iter<2:
            a = self._get_split_stepsize(self.x, grad, init_step=self.stepsize)
            self.xp = self.x[:]
            for i in range(len(self.x)):
                self.x[i] = self.x[i] - a[i] * grad[i]

            self.gradp = grad
        else:

        # independent BB step sizes
            for i in range(len(self.x)):
                grad_diff = grad[i] - self.gradp[i]
                self.stepsize[i] = abs(np.dot(self.x[i]-self.xp[i], grad_diff))/np.dot(grad_diff, grad_diff)
        # new solution: gradient descent step
            self.xp = self.x[:]
            for i in range(len(self.x)):
                self.x[i] = self.x[i] - self.stepsize[i]*grad[i]
            self.gradp = grad

        if self.callback is not None:
            self.callback(self)

        if self.bounds is not None:
            for i in range(len(self.x)):
                if self.bounds[i] is not None:
                    self.x[i] = np.clip(self.x[i], *self.bounds[i])

        if self.projector is not None:
            self.x = self.projector(self.x)

        self.iter +=1


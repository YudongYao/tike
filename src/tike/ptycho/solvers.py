"""This module provides Solver implementations for a variety of algorithms."""

from tike.opt import conjugate_gradient
from tike.ptycho import PtychoBackend

__all__ = [
    "available_solvers",
    "ConjugateGradientPtychoSolver",
]

class ConjugateGradientPtychoSolver(PtychoBackend):
    """Solve the ptychography problem using gradient descent."""

    def run(
        self, data, probe, scan, psi,
        reg=0j, num_iter=1, rho=0.0,
        model='poisson', recover_probe=False, dir_probe=None,
        **kwargs
    ):  # yapf: disable
        """Use conjugate gradient to estimate `psi`.

        Parameters
        ----------
        reg : (V, H, P) :py:class:`numpy.array` complex
            The regularizer for psi. (h + lamda / rho)
        rho : float
            The positive penalty parameter. It should be less than 1.

        """
        xp = self.array_module
        reg = xp.asarray(reg, 'complex64')

        if model is 'poisson':

            def maximum_a_posteriori_probability(farplane):
                simdata = xp.square(xp.abs(farplane))
                return xp.sum(simdata - data * xp.log(simdata + 1e-32))

            def data_diff(farplane):
                return farplane * (
                    1 - data / (xp.square(xp.abs(farplane)) + 1e-32))

        elif model is 'gaussian':

            def maximum_a_posteriori_probability(farplane):
                return xp.sum(xp.square(xp.abs(farplane) - xp.sqrt(data)))

            def data_diff(farplane):
                return (farplane
                        - xp.sqrt(data) * xp.exp(1j * xp.angle(farplane)))

        else:
            raise ValueError("model must be 'gaussian' or 'poisson.'")

        def cost_function(psi):
            farplane = self.fwd(psi=psi, scan=scan, probe=probe)
            return (
                + maximum_a_posteriori_probability(farplane)
                + rho * xp.square(xp.linalg.norm(reg - psi))
            )

        def grad(psi):
            farplane = self.fwd(psi=psi, scan=scan, probe=probe)
            grad_psi = self.adj(
                farplane=data_diff(farplane),
                probe=probe, scan=scan,
            )  # yapf: disable
            grad_psi /= xp.max(xp.abs(probe))**2  # this is not in the math
            grad_psi -= rho * (reg - psi)
            return grad_psi

        psi = conjugate_gradient(
            self.array_module,
            x=psi,
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

        # def get_grad_probe(farplane):
        #     grad_probe = self.adj_probe(
        #         farplane=data_diff(farplane),
        #         scan=scan,
        #         psi=psi,
        #     )  # yapf: disable
        #     grad_probe /= xp.square(xp.max(xp.abs(psi)))
        #     grad_probe /= self.nscan
        #
        # probe = ConjugateGradient.run(
        #     self,
        #     'probe',
        #     x=probe,
        #     num_iter=num_iter,
        #     cost_function=maximum_a_posteriori_probability,
        #     get_grad=get_grad,
        #     scan=scan,
        #     psi=psi,
        # )

        return {
            'psi': psi,
            'probe': probe,
        }


# TODO: Add new algorithms here
available_solvers = {
    "cgrad": ConjugateGradientPtychoSolver,
}
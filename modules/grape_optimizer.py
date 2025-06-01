import numpy as np
from scipy.optimize import minimize, basinhopping, dual_annealing, OptimizeResult
from concurrent.futures import ProcessPoolExecutor, as_completed

def _optimize_single_run(fidelity_func, grad_func, x0, method, options, threshold, maxiter, tol):
    """
    单次优化的辅助函数（用于多起始点并行时调用）。
    fidelity_func: 计算给定参数下门保真度的函数
    grad_func: 保真度对参数的梯度（可选）
    x0: 起始参数
    method: 优化方法
    options: 优化器选项（如 bounds, niter 等）
    threshold: 提前终止的 loss 阈值
    maxiter: 最大迭代次数
    tol: 收敛容限
    """
    x0 = np.array(x0, dtype=float)
    # 定义优化目标函数（loss = 1 - fidelity）
    def fun(x):
        return 1.0 - fidelity_func(x)
    # 定义梯度（如果提供）
    jacobian = None
    if method in ['L-BFGS-B', 'BFGS'] and grad_func is not None:
        jacobian = lambda x: -grad_func(x)
    # 提前停止检测：记录当前最优，并在低于阈值时抛出异常
    stop_flag = {"best_x": None, "best_val": np.inf}
    def callback(xk):
        val = fun(xk)
        if val < stop_flag["best_val"]:
            stop_flag["best_val"] = val
            stop_flag["best_x"] = xk.copy()
        if val < threshold:
            raise StopIteration

    try:
        if method == 'basinhopping':
            # 基于跳跃的全局优化
            minimizer_kwargs = {"method": "L-BFGS-B", "jac": jacobian}
            res = basinhopping(fun, x0, minimizer_kwargs=minimizer_kwargs,
                               niter=options.get('niter', 100))
        elif method == 'dual_annealing':
            # 双退火全局优化
            bounds = options.get('bounds', None)
            res = dual_annealing(fun, bounds=bounds, maxiter=options.get('maxiter', 1000))
        else:
            # 局部优化（BFGS / L-BFGS-B 等）
            opt_opts = {"maxiter": maxiter}
            # 设置收敛精度
            if tol is not None:
                if method == 'L-BFGS-B':
                    opt_opts['ftol'] = tol
                elif method == 'BFGS':
                    opt_opts['gtol'] = tol
            opt_opts.update(options)
            res = minimize(fun, x0, method=method, jac=jacobian,
                           options=opt_opts,
                           callback=callback if grad_func is not None else None)
    except StopIteration:
        # 达到阈值提前终止，提取当前最优解
        best_x = stop_flag["best_x"]
        best_val = stop_flag["best_val"]
        res = OptimizeResult(x=best_x, fun=best_val, success=True,
                             message=f"阈值达到: loss={best_val:.3g} < {threshold}")
    return res

class GRAPEOptimizer:
    """
    GRAPE 优化器类，用于优化脉冲参数以实现高保真量子门。

    参数:
        fidelity_func (callable): 输入参数数组，输出门保真度（0~1）。
        initial_params (ndarray): 初始脉冲参数猜测（1D 数组）。
        grad_func (callable, 可选): fidelity_func 对参数的梯度函数。
    """
    def __init__(self, fidelity_func, initial_params, grad_func=None):
        self.fidelity_func = fidelity_func
        self.grad_func = grad_func
        self.initial_params = np.array(initial_params, dtype=float)
        self.best_result = None

    def optimize(self, method='L-BFGS-B', multi_start=1, maxiter=1000, tol=None,
                 threshold=1e-3, random_seed=None, parallel=True, **options):
        """
        运行优化过程。

        参数:
            method (str): 优化方法，例如 'L-BFGS-B', 'BFGS', 'basinhopping', 'dual_annealing' 等。
            multi_start (int): 随机起始点数量，用于 multi-start 策略。
            maxiter (int): 单次局部优化的最大迭代次数。
            tol (float): 收敛容限（精度）。
            threshold (float): loss 阈值，低于该值时提前停止。
            random_seed (int): 随机种子，用于复现随机初始参数。
            parallel (bool): 是否并行运行多次起始点优化。
            **options: 其他优化器参数，例如对 dual_annealing 可设置 bounds，对 basinhopping 可设置 niter 等。
        返回:
            OptimizeResult: 最优结果对象，包含优化后的参数、loss 和状态信息。
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        base = self.initial_params
        # 生成多组初始参数
        inits = []
        for i in range(multi_start):
            if i == 0:
                inits.append(base.copy())
            else:
                # 加入随机扰动
                perturb = np.random.normal(scale=0.1, size=base.shape)
                inits.append(base + perturb)
        results = []
        # 并行优化多个起始点
        if parallel and multi_start > 1:
            futures = []
            with ProcessPoolExecutor() as executor:
                for x0 in inits:
                    args = (self.fidelity_func, self.grad_func, x0,
                            method, options, threshold, maxiter, tol)
                    futures.append(executor.submit(_optimize_single_run, *args))
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            # 顺序优化
            for x0 in inits:
                res = _optimize_single_run(self.fidelity_func, self.grad_func, x0,
                                           method, options, threshold, maxiter, tol)
                results.append(res)
        # 从所有结果中选取 loss 最小的解
        best = min(results, key=lambda res: res.fun)
        self.best_result = best
        return best

# 示例用法（运行本模块时执行）
if __name__ == "__main__":
    import numpy as np
    # 定义一个测试用保真度函数：fidelity = exp(-||params||^2)
    def dummy_fidelity(x):
        return np.exp(-np.sum(x**2))
    init_params = np.random.rand(5)  # 随机初始参数
    optimizer = GRAPEOptimizer(dummy_fidelity, initial_params=init_params)
    # 执行优化：使用 L-BFGS-B，多起始点，允许早停
    result = optimizer.optimize(method='L-BFGS-B', multi_start=3, maxiter=200,
                                tol=1e-6, threshold=1e-3, random_seed=123)
    print("最优 loss:", result.fun, "对应 fidelity:", 1 - result.fun)

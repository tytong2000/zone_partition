import gurobipy as gp
from gurobipy import GRB
from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution

class GurobiSolver(BaseSolver):
    def __init__(self, vrp_instance):
        """
        使用 Gurobi 求解 VRP 问题
        
        :param vrp_instance: VRP 实例
        """
        self.vrp_instance = vrp_instance

    def solve(self):
        """
        使用 Gurobi 求解 VRP 问题
        """
        # 创建 Gurobi 模型
        model = gp.Model("VRP")
        
        # 定义变量：x[i,j] 为路径是否从 i 到 j
        x_vars = {}
        for i in range(self.vrp_instance.num_customers):
            for j in range(self.vrp_instance.num_customers):
                if i != j:
                    x_vars[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        # 设置目标函数: 最小化总路程
        objective = gp.LinExpr()
        for i in range(self.vrp_instance.num_customers):
            for j in range(self.vrp_instance.num_customers):
                if i != j:
                    objective += self.vrp_instance.distances[i][j] * x_vars[i, j]
        
        model.setObjective(objective, GRB.MINIMIZE)
        
        # 添加约束: 每个客户必须进入一次且离开一次
        for i in range(self.vrp_instance.num_customers):
            model.addConstr(gp.quicksum(x_vars[i, j] for j in range(self.vrp_instance.num_customers) if i != j) == 1)
            model.addConstr(gp.quicksum(x_vars[j, i] for j in range(self.vrp_instance.num_customers) if i != j) == 1)
        
        # 求解模型
        model.optimize()

        if model.status == GRB.OPTIMAL:
            solution = {}
            for i in range(self.vrp_instance.num_customers):
                for j in range(self.vrp_instance.num_customers):
                    if i != j and x_vars[i, j].x > 0.5:
                        solution[i] = j
            return solution, model.objVal
        else:
            return None, None

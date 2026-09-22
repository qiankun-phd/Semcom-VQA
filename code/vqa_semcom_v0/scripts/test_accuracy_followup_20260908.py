import unittest
import numpy as np
import accuracy_followup_20260908_ab as a

class FollowupTests(unittest.TestCase):
    def test_logistic_gradient(self):
        rng=np.random.default_rng(7);x=np.column_stack([rng.normal(size=(20,3)),np.ones(20)]);y=rng.integers(0,2,(20,2));w=rng.normal(size=8)*.1
        _,g=a.logistic_objective(w,x,y)
        numerical=[]
        for i in range(8):
            d=np.zeros(8);d[i]=1e-6;numerical.append((a.logistic_objective(w+d,x,y)[0]-a.logistic_objective(w-d,x,y)[0])/2e-6)
        np.testing.assert_allclose(g,numerical,atol=1e-8)
    def test_historical_optimizer_and_ridge(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(60,4));y=rng.integers(0,2,(60,2)).astype(float)
        fits=a.fit_linears(x,y,True)
        np.testing.assert_array_equal(fits['logistic_gd400']['weights'],a.original.fit_linear(x,y))
        self.assertLess(fits['ridge_advantage']['gradient_inf_norm'],1e-8)
        self.assertLessEqual(fits['logistic_converged']['train_objective'],fits['logistic_gd400']['train_objective']+1e-8)
    def test_common_target_infeasible(self):
        self.assertIsNone(a.select([{'accuracy':.70,'image_fraction':.1,'penalty':0}],.74))
    def test_same_resource_policy(self):
        r=[{'accuracy':.8,'image_fraction':.5,'penalty':0},{'accuracy':.75,'image_fraction':.1,'penalty':.2},{'accuracy':.7,'image_fraction':0,'penalty':1}]
        self.assertEqual(a.select(r,.74),1)
    def test_kappa_units_and_tie(self):
        y=np.array([[0,1],[1,0]]);s=np.array([.5,0]);records,picks=a.sweep(y,s,[0,.5])
        np.testing.assert_array_equal(picks,[[1,0],[0,0]])
        self.assertNotIn('energy_j_original_accounting',records[0])

if __name__=='__main__':unittest.main()

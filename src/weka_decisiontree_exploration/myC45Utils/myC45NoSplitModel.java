/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myC45NoSplitModel extends myC45ClassifierSplitModel {
    public myC45NoSplitModel(myDistribution distribution) {
        dist = distribution;
        nbSubset = 1;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        dist = new myDistribution(instances);
        nbSubset = 1;
    }

    @Override
    public double[] getWeights(Instance instance) {
        return null;
    }

    @Override
    public int getSubset(Instance instance) throws Exception {
        return 0;
    }
}

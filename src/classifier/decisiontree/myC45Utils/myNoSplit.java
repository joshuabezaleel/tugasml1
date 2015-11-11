/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.decisiontree.myC45Utils;

import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class myNoSplit extends myClassifierSplitModel {

    public myNoSplit(myDistribution distribution) {
        dist = new myDistribution(distribution);
        nbSubset = 1;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        dist = new myDistribution(instances);
        nbSubset = 1;
    }

    @Override
    public int whichSubset(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] weights(Instance instance) {
        return null;
    }
    
}

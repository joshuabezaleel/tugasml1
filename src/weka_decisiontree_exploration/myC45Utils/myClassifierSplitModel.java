/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public abstract class myClassifierSplitModel implements Serializable {
    public myDistribution dist;
    protected int nbSubset;
    
    public abstract void buildClassifier(Instances instances) throws Exception;
    public abstract int whichSubset(Instance instance) throws Exception;

    public int numSubsets() {
        return nbSubset;
    }
    
    public final boolean checkModel() {
        if (nbSubset > 0)
            return true;
        else
            return false;
  }

    public Instances[] split(Instances data) throws Exception {
        Instances [] instances = new Instances [nbSubset];
        double [] weights;
        double newWeight;
        Instance instance;
        int subset, i, j;
        
        for (j=0; j<nbSubset; j++)
            instances[j] = new Instances((Instances)data, data.numInstances());
        for (i = 0; i < data.numInstances(); i++) {
            instance = ((Instances) data).instance(i);
            weights = weights(instance);
            subset = whichSubset(instance);
            if (subset > -1)
                instances[subset].add(instance);
            else
                for (j = 0; j < nbSubset; j++)
                    if (Utils.gr(weights[j],0)) {
                        newWeight = weights[j]*instance.weight();
                        instances[j].add(instance);
                        instances[j].lastInstance().setWeight(newWeight);
                    }
        }
        for (j = 0; j < nbSubset; j++)
            instances[j].compactify();
        
        return instances;
    }

    public myDistribution distribution() {
        return dist;
    }

    public void resetDistribution(Instances instances) throws Exception {
        dist = new myDistribution(instances, this);
    }

    public abstract double [] weights(Instance instance);

    public double classProb(int classIndex, Instance instance, int theSubset) {
        if (theSubset > -1) {
            return dist.prob(classIndex,theSubset);
        } else {
            double [] weights = weights(instance);
            if (weights == null) {
                return dist.prob(classIndex);
            } else {
                double prob = 0;
                for (int i = 0; i < weights.length; i++) {
                    prob += weights[i] * dist.prob(classIndex, i);
                }
                return prob;
            }
        }
    }
}

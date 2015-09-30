/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public abstract class myC45ClassifierSplitModel {
    public myDistribution dist;
    protected int nbSubset;
    
    public abstract void buildClassifier(Instances instances) throws Exception;
    public abstract double [] Weights(Instance instance);
    public abstract int whichSubset(Instance instance) throws Exception;
    public final int nbSubset() {
        return nbSubset;
    }
    public final Instances [] split(Instances data) throws Exception {
        Instances [] instances = new Instances [nbSubset];
        double [] weights;
        double newWeight;
        Instance instance;
        int subset;
        
        for (int i = 0; i < nbSubset; i++) {
            instances[i] = new Instances(data, data.numInstances());
        }
        for(int j = 0; j < data.numInstances(); j++) {
            instance = data.instance(j);
            weights = Weights(instance);
            subset = whichSubset(instance);
            if(subset > -1) {
                instances[subset].add(instance);
            }
            else {
                for(int i = 0; i < nbSubset; i++) {
                    if(Utils.gr(weights[i], 0)) {
                        newWeight = weights[i]*instance.weight();
                        instances[i].add(instance);
                        instances[i].lastInstance().setWeight(newWeight);
                    }
                }
            }
        }
        for (int i = 0; i < nbSubset; i++) {
            instances[i].compactify();
        }
        return instances;
    }
    public final boolean isValidModel() {
        return (nbSubset > 0);
    }
    public myDistribution getDist() {
        return dist;
    }
    public void resetDist(Instances data) throws Exception {
        dist = new myDistribution(data, this);
    }
    public double getClassProbability(int idxClass, Instance instance, int subset) {
        if(subset > -1) {
            return dist.prob(idxClass, subset);
        }
        else {
            double [] weights = Weights(instance);
            if(weights == null) {
                return dist.prob(idxClass);
            }
            else {
                double probability = 0;
                for(int i = 0; i < weights.length; i++) {
                    probability += weights[i] * dist.prob(idxClass, i);
                }
                return probability;
            }
        }
    }
}

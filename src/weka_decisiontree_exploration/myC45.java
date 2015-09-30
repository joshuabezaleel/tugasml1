/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Statistics;
import weka.core.Utils;
import weka_decisiontree_exploration.myC45Utils.myC45ClassifierSplitModel;
import weka_decisiontree_exploration.myC45Utils.myC45NoSplitModel;
import weka_decisiontree_exploration.myC45Utils.myC45SplitModel;
import weka_decisiontree_exploration.myC45Utils.myDistribution;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myC45 extends Classifier {
    
    private myC45[] successor;
    private myC45ClassifierSplitModel local_model;
    private boolean is_leaf;
    private boolean is_empty;
    private boolean use_pruning = false;
    private Instances m_train;
    private myDistribution m_test;
    private boolean m_isLeaf;
    private boolean m_isEmpty;
    
    public void setPruning(boolean prune) {
        use_pruning = prune;
    }    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        Instances data = new Instances(i);
        data.deleteWithMissingClass();
        buildTree(data, true);
        collapse();
        prune();
    }    
    public void buildTree(Instances data, boolean keepData) throws Exception {
        Instances [] localInstances;

        if (keepData) {
          m_train = data;
        }
        m_test = null;
        m_isLeaf = false;
        m_isEmpty = false;
        successor = null;
        local_model = selectModel(data, data);
        if(local_model.nbSubset() > 1) {
            localInstances = local_model.split(data);
            data = null;
            successor = new myC45[local_model.nbSubset()];
            for(int i = 0; i < successor.length; i++) {
                successor[i] = getNewTree(localInstances[i]);
                localInstances[i] = null;
            }
        }
        else {
            m_isLeaf = true;
            if(Utils.eq(data.sumOfWeights(), 0)) {
                m_isEmpty = true;
                data = null;
            }
        }
    }    
    public void collapse() {
        double errorsOfSubtree;
        double errorsOfTree;
        
        if(!m_isLeaf) {
            errorsOfSubtree = getTrainingError();
            errorsOfTree = getLocalModel().getDist().numIncorrect();
            if(errorsOfSubtree >= errorsOfTree-1E-3) {
                successor = null;
                m_isLeaf = true;
                local_model = new myC45NoSplitModel(getLocalModel().getDist());
            }
            else {
                for(int i = 0; i < successor.length; i++) {
                    successor[i].collapse();
                }
            }
        }
    }
    public double addErrs(double N, double e, float CF){
        if (CF > 0.5) {
            System.err.println("WARNING: confidence value for pruning too high. Error estimate not modified.");
            return 0;
        }
        if (e < 1) {
            double base = N * (1 - Math.pow(CF, 1 / N)); 
            if (e == 0) {
                return base; 
            }
            return base + e * (addErrs(N, 1, CF) - base);
        }
        if (e + 0.5 >= N) {
            return Math.max(N - e, 0);
        }
        double z = Statistics.normalInverse(1 - CF);
        double  f = (e + 0.5) / N;
        double r = (f + (z * z) / (2 * N) + z * Math.sqrt((f / N) -  (f * f / N) +  (z * z / (4 * N * N)))) / (1 + (z * z) / N);
        return (r * N) - e;
    }
    public double getEstimatedErrorsForDistribution(myDistribution theDistribution) {
        if (Utils.eq(theDistribution.total(),0))
            return 0;
        else
            return theDistribution.numIncorrect()+addErrs(theDistribution.total(),theDistribution.numIncorrect(),0.25f);
    }
    public double getEstimatedErrorsForBranch(Instances data) throws Exception {
        Instances [] localInstances;
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(new myDistribution(data));
        else{
            myDistribution savedDist = getLocalModel().getDist();
            getLocalModel().resetDist(data);
            localInstances = (Instances[])getLocalModel().split(data);
            getLocalModel().dist = savedDist;
            for (i=0;i<successor.length;i++) {
                errors = errors+successor[i].getEstimatedErrorsForBranch(localInstances[i]);
            }
            return errors;
        }
    }
    public void prune() throws Exception {
        double errorsLargestBranch;
        double errorsLeaf;
        double errorsTree;
        int indexOfLargestBranch;
        myC45 largestBranch;
        int i;

        if (!m_isLeaf){
            for (i=0;i<successor.length;i++) {
                successor[i].prune();
            }
            indexOfLargestBranch = getLocalModel().getDist().maxBag();
            errorsLargestBranch = successor[indexOfLargestBranch].getEstimatedErrorsForBranch((Instances)m_train);
            errorsLeaf = getEstimatedErrorsForDistribution(getLocalModel().getDist());
            errorsTree = getEstimatedErrors();
            if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) && Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)) {
                successor = null;
                m_isLeaf = true;
                local_model = new myC45NoSplitModel(getLocalModel().getDist());
                return;
            }
            if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)) {
                largestBranch = successor[indexOfLargestBranch];
                successor = largestBranch.successor;
                local_model = largestBranch.getLocalModel();
                m_isLeaf = largestBranch.m_isLeaf;
                newDistribution(m_train);
                prune();
            }
        }
    }
    public void newDistribution(Instances data) throws Exception {
        Instances [] localInstances;
        
        getLocalModel().resetDist(data);
        m_train = data;
        if (!m_isLeaf){
            localInstances = (Instances [])getLocalModel().split(data);
            for (int i = 0; i < successor.length; i++)
                successor[i].newDistribution(localInstances[i]);
        } else {
            if (!Utils.eq(data.sumOfWeights(), 0)) {
                m_isEmpty = false;
            }
        }
    }
    public double getTrainingError() {
        double error = 0;
        
        if(m_isLeaf) {
            return getLocalModel().getDist().numIncorrect();
        }
        else {
            for(int i = 0; i < successor.length; i++) {
                error += successor[i].getTrainingError();
            }
            return error;
        }
    }
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double maxProb = -1;
        double currentProb;
        int idxMax = 0;
        
        for(int i = 0; i < instance.numClasses(); i++) {
            currentProb = getProbabilities(i, instance, 1);
            if(Utils.gr(currentProb, maxProb)) {
                idxMax = i;
                maxProb = currentProb;
            }
        }
        return (double) idxMax;
    }
    public myC45 getNewTree(Instances data) throws Exception {
        myC45 newTree = new myC45();
        newTree.buildTree(data, false);
        return newTree;
    }
    public double getProbabilities(int idxClass, Instance instance, double weight) throws Exception {
        double prob = 0;
        if(is_leaf) {
            return weight * getLocalModel().getClassProbability(idxClass, instance, -1);
        }
        else {
            int idxTree = getLocalModel().whichSubset(instance);
            if(idxTree == -1) {
                double [] weights = getLocalModel().Weights(instance);
                for(int i = 0; i < successor.length; i++) {
                    if(!successor[i].is_empty) {
                        prob += successor[i].getProbabilities(idxClass, instance, weights[i] * weight);
                    }
                }
                return prob;
            }
            else {
                if(successor[idxTree].is_empty) {
                    return weight * getLocalModel().getClassProbability(idxClass, instance, idxTree);
                }
                else {
                    return successor[idxTree].getProbabilities(idxClass, instance, weight);
                }
            }
        }
    }
    public myC45ClassifierSplitModel getLocalModel() {
        return (myC45ClassifierSplitModel) local_model;
    }
    public myC45ClassifierSplitModel selectModel(Instances alldata, Instances data) throws Exception {
        double minResult;
        double currentResult;
        myC45SplitModel [] currentModel;
        myC45SplitModel bestModel = null;
        myC45NoSplitModel noSplitModel = null;
        double averageInfoGain = 0;
        int validModels = 0;
        boolean multiVal = true;
        myDistribution checkDistribution;
        Attribute attribute;
        double sumOfWeights;
        
        checkDistribution = new myDistribution(data);
        noSplitModel = new myC45NoSplitModel(checkDistribution);
        if(Utils.sm(checkDistribution.total(), 4) || Utils.eq(checkDistribution.total(), checkDistribution.perClass(checkDistribution.maxClass()))) {
            return noSplitModel;
        }
        if(alldata != null) {
            Enumeration en = data.enumerateAttributes();
            while(en.hasMoreElements()) {
                attribute = (Attribute) en.nextElement();
                if(attribute.isNumeric() || Utils.sm((double)attribute.numValues(), 0.3*(double)alldata.numInstances())) {
                    multiVal = false;
                    break;
                }
            }
        }
        currentModel = new myC45SplitModel[data.numAttributes()];
        sumOfWeights = data.sumOfWeights();
        for(int i = 0; i < data.numAttributes(); i++) {
            if(i != (data).classIndex()) {
                currentModel[i] = new myC45SplitModel(i,2,sumOfWeights);
                currentModel[i].buildClassifier(data);
                if(currentModel[i].checkModel()) {
                    if(alldata != null) {
                        if(data.attribute(i).isNumeric() || (multiVal || Utils.sm((double)data.attribute(i).numValues(), 0.3*(double)alldata.numInstances()))) {
                            averageInfoGain += currentModel[i].infoGain();
                            validModels++;
                        }
                    }
                    else {
                        averageInfoGain += currentModel[i].infoGain();
                        validModels++;
                    }
                }
            }
            else {
                currentModel[i] = null;
            }
        }
        if(validModels == 0) {
            return noSplitModel;
        }
        averageInfoGain = averageInfoGain / (double)validModels;
        minResult = 0;
        for(int i = 0; i < data.numAttributes(); i++) {
            if((i != (data).classIndex()) && currentModel[i].checkModel()) {
                if((currentModel[i].infoGain() >= (averageInfoGain-1E-3)) && Utils.gr(currentModel[i].gainRatio(), minResult)) {
                    bestModel = currentModel[i];
                    minResult = currentModel[i].gainRatio();
                }
            }
        }
        if(Utils.eq(minResult, 0)) {
            return noSplitModel;
        }
        bestModel.getDist().addInstWithUnknown(data,bestModel.attIndex());
        if(alldata != null) {
            bestModel.setSplitPoint(alldata);
        }
        return bestModel;
    }

    public double getEstimatedErrors() {
        double errors = 0;
        
        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(getLocalModel().getDist());
        else{
            for (int i = 0; i < successor.length;i++)
                errors = errors+successor[i].getEstimatedErrors();
            return errors;
        }
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Statistics;
import weka.core.Utils;
import weka_decisiontree_exploration.myC45Utils.myClassifierSplitModel;
import weka_decisiontree_exploration.myC45Utils.myDistribution;
import weka_decisiontree_exploration.myC45Utils.myModelSelection;
import weka_decisiontree_exploration.myC45Utils.myNoSplit;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class myC45 extends Classifier {
    private float m_CF = 0.25f;
    private Instances m_train;
    private myDistribution m_test;
    private myModelSelection m_toSelectModel;
    private myClassifierSplitModel m_localModel;
    private boolean m_isLeaf;
    private boolean m_isEmpty;
    protected myC45[] m_sons;

    public myC45() {
        super();
    }

    public myC45(myModelSelection toSelectModel) {
        m_toSelectModel = toSelectModel;
    }

    private void buildTree(Instances data) throws Exception {
        Instances [] localInstances;

        m_train = data;
        m_test = null;
        m_isLeaf = false;
        m_isEmpty = false;
        m_sons = null;
        m_localModel = m_toSelectModel.selectModel(data);
        if (m_localModel.numSubsets() > 1) {
            localInstances = m_localModel.split(data);
            data = null;
            m_sons = new myC45 [m_localModel.numSubsets()];
            for (int i = 0; i < m_sons.length; i++) {
                m_sons[i] = getNewTree(localInstances[i]);
                localInstances[i] = null;
            }
        }else{
            m_isLeaf = true;
            if (Utils.eq(data.sumOfWeights(), 0))
                m_isEmpty = true;
            data = null;
        }
    }

    private void collapse() {
        double errorsOfSubtree;
        double errorsOfTree;
        int i;

        if (!m_isLeaf){
            errorsOfSubtree = getTrainingErrors();
            errorsOfTree = localModel().distribution().numIncorrect();
            if (errorsOfSubtree >= errorsOfTree-1E-3){
                // Free adjacent trees
                m_sons = null;
                m_isLeaf = true;
                
                // Get NoSplit Model for tree.
                m_localModel = new myNoSplit(localModel().distribution());
            }else
                for (i=0;i<m_sons.length;i++)
                    son(i).collapse();
        }
    }

    private void prune() throws Exception {
        double errorsLargestBranch;
        double errorsLeaf;
        double errorsTree;
        int indexOfLargestBranch;
        myC45 largestBranch;
        int i;
        
        if (!m_isLeaf){
            // Prune all subtrees.
            for (i=0;i<m_sons.length;i++)
                son(i).prune();
            // Compute error for largest branch
            indexOfLargestBranch = localModel().distribution().maxBag();
            errorsLargestBranch = son(indexOfLargestBranch).getEstimatedErrorsForBranch((Instances)m_train);
            
            // Compute error if this Tree would be leaf
            errorsLeaf = getEstimatedErrorsForDistribution(localModel().distribution());
            // Compute error for the whole subtree
            errorsTree = getEstimatedErrors();
            // Decide if leaf is best choice.
            if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) && Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){
                // Free son Trees
                m_sons = null;
                m_isLeaf = true;
                
                // Get NoSplit Model for node.
                m_localModel = new myNoSplit(localModel().distribution());
                return;
            }
            // Decide if largest branch is better choice
            // than whole subtree.
            if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
                largestBranch = son(indexOfLargestBranch);
                m_sons = largestBranch.m_sons;
                m_localModel = largestBranch.localModel();
                m_isLeaf = largestBranch.m_isLeaf;
                newDistribution(m_train);
                prune();
            }
        }
    }

    private void cleanup(Instances instances) {
        m_train = instances;
        m_test = null;
        if (!m_isLeaf)
            for (int i = 0; i < m_sons.length; i++)
                m_sons[i].cleanup(instances);
    }

    private myC45 getNewTree(Instances localInstance) throws Exception {
        myC45 newTree = new myC45(m_toSelectModel);
        newTree.buildTree((Instances)localInstance);

        return newTree;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(m_toSelectModel == null)
            m_toSelectModel = new myModelSelection(2, data);
        
        data = new Instances(data);
        data.deleteWithMissingClass();
        buildTree(data);
        collapse();
        prune();
        cleanup(new Instances(data, 0));
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double maxProb = -1;
        double currentProb;
        int maxIndex = 0;
        int j;

        for (j = 0; j < instance.numClasses(); j++) {
            currentProb = getProbs(j, instance, 1);
            if (Utils.gr(currentProb,maxProb)) {
                maxIndex = j;
                maxProb = currentProb;
            }
        }

        return (double)maxIndex;
    }

    private double getTrainingErrors() {
        double errors = 0;
        int i;

        if (m_isLeaf)
            return localModel().distribution().numIncorrect();
        else {
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getTrainingErrors();
            return errors;
        }
    }

    private myClassifierSplitModel localModel() {
        return (myClassifierSplitModel) m_localModel;
    }

    private myC45 son(int i) {
        return (myC45)m_sons[i];
    }

    private double getEstimatedErrorsForBranch(Instances instances) throws Exception {
        Instances [] localInstances;
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(new myDistribution(instances));
        else {
            myDistribution savedDist = localModel().dist;
            localModel().resetDistribution(instances);
            localInstances = (Instances[])localModel().split(instances);
            localModel().dist = savedDist;
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getEstimatedErrorsForBranch(localInstances[i]);
            return errors;
        }
    }

    private double getEstimatedErrorsForDistribution(myDistribution distribution) {
        if (Utils.eq(distribution.total(),0))
            return 0;
        else
            return distribution.numIncorrect()+addErrs(distribution.total(), distribution.numIncorrect(), m_CF);
    }

    private double getEstimatedErrors() {
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(localModel().distribution());
        else {
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getEstimatedErrors();
            return errors;
        }
    }

    private void newDistribution(Instances m_train) throws Exception {
        Instances [] localInstances;

        localModel().resetDistribution(m_train);
        m_train = m_train;
        if (!m_isLeaf){
            localInstances = (Instances [])localModel().split(m_train);
            for (int i = 0; i < m_sons.length; i++)
                son(i).newDistribution(localInstances[i]);
        } else {
            // Check whether there are some instances at the leaf now!
            if (!Utils.eq(m_train.sumOfWeights(), 0)) {
                m_isEmpty = false;
            }
        }
    }

    private double addErrs(double N, double e, float CF) {
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

    private double getProbs(int classIndex, Instance instance, double weight) throws Exception {
        double prob = 0;
    
        if (m_isLeaf) {
            return weight * localModel().classProb(classIndex, instance, -1);
        } else {
            int treeIndex = localModel().whichSubset(instance);
            if (treeIndex == -1) {
                double[] weights = localModel().weights(instance);
                for (int i = 0; i < m_sons.length; i++) {
                    if (!son(i).m_isEmpty) {
                        prob += son(i).getProbs(classIndex, instance, weights[i] * weight);
                    }
                }
                return prob;
            } else {
                if (son(treeIndex).m_isEmpty) {
                    return weight * localModel().classProb(classIndex, instance, treeIndex);
                } else {
                    return son(treeIndex).getProbs(classIndex, instance, weight);
                }
            }
        }
    }
    
}

package weka_decisiontree_exploration.myC45Utils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

public class myC45SplitModel extends myC45ClassifierSplitModel{

  private int m_attIndex;         
  private int m_minNoObj;         
  private double m_splitPoint;   
  private double m_infoGain; 
  private double m_gainRatio;  
  private double m_sumOfWeights;  
  private int m_complexityIndex;  
  private int m_index;            

  //private static InfoGainSplitCrit infoGainCrit = new InfoGainSplitCrit();
  //private static GainRatioSplitCrit gainRatioCrit = new GainRatioSplitCrit();


  public myC45SplitModel(int attIndex,int minNoObj, double sumOfWeights) {
    m_attIndex = attIndex;
    m_minNoObj = minNoObj;
    m_sumOfWeights = sumOfWeights;
  }

  public void buildClassifier(Instances training) throws Exception {
    nbSubset = 0;
    m_splitPoint = Double.MAX_VALUE;
    m_infoGain = 0;
    m_gainRatio = 0;

    //Jika datanya nominal
    if (training.attribute(m_attIndex).isNominal()) {
      m_complexityIndex = training.attribute(m_attIndex).numValues();
      m_index = m_complexityIndex;
      handleEnumeratedAttribute(training);
    } else { //Jika datanya numerik
      m_complexityIndex = 2;
      m_index = 0;
      training.sort(training.attribute(m_attIndex));
      handleNumericAttribute(training);
    }
  }

  public final boolean checkModel() {
    if (nbSubset > 0){
      return true;
    } else {
      return false;
    }
  }
  
  public final double infoGain() {
    return m_infoGain;
  }

  public final double gainRatio() {
    return m_gainRatio;
  }

  public final void distribution() {

  }

  public final int attIndex() {
    return m_attIndex;
  }


  public final void setSplitPoint(Instances data){
    double newSplitPoint = -Double.MAX_VALUE;
    double tempValue;
    Instance instance;
    
    if ((data.attribute(m_attIndex).isNumeric()) && (nbSubset > 1)) {
      Enumeration enu = data.enumerateInstances();
      while (enu.hasMoreElements()) {
        instance = (Instance) enu.nextElement();
        if (!instance.isMissing(m_attIndex)) {
          tempValue = instance.value(m_attIndex);
            if (Utils.gr(tempValue,newSplitPoint) && 
                Utils.smOrEq(tempValue,m_splitPoint))
                newSplitPoint = tempValue;
            }
        }
      m_splitPoint = newSplitPoint;
    }
  }

  public final double classProb(int classIndex,Instance instance, int theSubset) throws Exception {
    if (theSubset <= -1) {
      double [] weights = Weights(instance);
      if (weights == null) {
        return dist.prob(classIndex);
      } else {
        double prob = 0;
        for (int i = 0; i < weights.length; i++) {
          prob += weights[i] * dist.prob(classIndex, i);
        }
        return prob;
      }
    } else {
      if (Utils.gr(dist.perBag(theSubset), 0)) {
        return dist.prob(classIndex, theSubset);
      } else {
        return dist.prob(classIndex);
      }
    }
  }

  //Menangani atribut nominal
  private void handleEnumeratedAttribute(Instances trainInstances) throws Exception {
    
    Instance instance;

    dist = new myDistribution(m_complexityIndex, trainInstances.numClasses());
    
    Enumeration enu = trainInstances.enumerateInstances();
    while (enu.hasMoreElements()) {
      instance = (Instance) enu.nextElement();
      if (!instance.isMissing(m_attIndex))
      {
        dist.add((int)instance.value(m_attIndex),instance);
      }
    }
    
    if (dist.check(m_minNoObj)) {
      nbSubset = m_complexityIndex;
      m_infoGain = infoGainCrit.splitCritValue(dist,m_sumOfWeights);
      m_gainRatio = gainRatioCrit.splitCritValue(dist, m_sumOfWeights, m_infoGain);
    }
  }

  //Menangani atribut numerik
  private void handleNumericAttribute(Instances trainInstances) throws Exception {
  
    int firstMiss;
    int next = 1;
    int last = 0;
    int splitIndex = -1;
    double currentInfoGain;
    double defaultEnt;
    double minSplit;
    Instance instance;
    int i;

    dist = new myDistribution(2,trainInstances.numClasses());
    
    Enumeration enu = trainInstances.enumerateInstances();
    i = 0;
    while (enu.hasMoreElements()) {
      instance = (Instance) enu.nextElement();
      if (instance.isMissing(m_attIndex))
    break;
      dist.add(1,instance);
      i++;
    }
    firstMiss = i;
    
    // minimal instance yang dibutuhkan
    minSplit =  0.1*(dist.total())/((double)trainInstances.numClasses());
    if (Utils.smOrEq(minSplit,m_minNoObj)) {
      minSplit = m_minNoObj;
    } else {
      if (Utils.gr(minSplit,25)) 
        minSplit = 25;
    }
    
    defaultEnt = infoGainCrit.oldEnt(dist);
    while (next < firstMiss) {
      if (trainInstances.instance(next-1).value(m_attIndex)+1e-5 < trainInstances.instance(next).value(m_attIndex)) { 
        dist.shiftRange(1,0,trainInstances,last,next);    

        if (Utils.grOrEq(dist.perBag(0),minSplit) && Utils.grOrEq(dist.perBag(1),minSplit)) {
          currentInfoGain = infoGainCrit.
          splitCritValue(dist, m_sumOfWeights, defaultEnt);
          if (Utils.gr(currentInfoGain,m_infoGain)) {
            m_infoGain = currentInfoGain;
            splitIndex = next-1;
          }
          m_index++;
        }

        last = next;
      }
      next++;
    }
    
    if (m_index == 0)
    {
      return;
    }
      
    
    m_infoGain = m_infoGain-(Utils.log2(m_index)/m_sumOfWeights);
    if (Utils.smOrEq(m_infoGain,0)) {
      return;
    }
      
    nbSubset = 2;
    m_splitPoint = (trainInstances.instance(splitIndex+1).value(m_attIndex) + trainInstances.instance(splitIndex).value(m_attIndex))/2;

    if (m_splitPoint == trainInstances.instance(splitIndex + 1).value(m_attIndex)) {
      m_splitPoint = trainInstances.instance(splitIndex).value(m_attIndex);
    }

    dist = new myDistribution(2,trainInstances.numClasses());
    dist.addRange(0,trainInstances,0,splitIndex+1);
    dist.addRange(1,trainInstances,splitIndex+1,firstMiss);

    m_gainRatio = gainRatioCrit.splitCritValue(dist, m_sumOfWeights, m_infoGain);
  }

  public final String leftSide(Instances data) {

    return data.attribute(m_attIndex).name();
  }

  public final String rightSide(int index,Instances data) {

    StringBuffer text;

    text = new StringBuffer();
    if (data.attribute(m_attIndex).isNominal()) {
      text.append(" = "+data.attribute(m_attIndex).value(index));
    } else {
      if (index == 0) {
        text.append(" <= "+Utils.doubleToString(m_splitPoint,6));    
      } else {
        text.append(" > "+Utils.doubleToString(m_splitPoint,6));
      }
    }
  
    return text.toString();
  }

  public final double [][] minsAndMaxs(Instances data, double [][] minsAndMaxs, int index) {

    double [][] newMinsAndMaxs = new double[data.numAttributes()][2];
    
    for (int i = 0; i < data.numAttributes(); i++) {
      newMinsAndMaxs[i][0] = minsAndMaxs[i][0];
      newMinsAndMaxs[i][1] = minsAndMaxs[i][1];
      if (i == m_attIndex)
      {
        if (data.attribute(m_attIndex).isNominal()){
          newMinsAndMaxs[m_attIndex][1] = 1;    
        } else {
          newMinsAndMaxs[m_attIndex][1-index] = m_splitPoint;    
        }
      }
    }

    return newMinsAndMaxs;
  }
  
  public void resetDistribution(Instances data) throws Exception {
    
    Instances insts = new Instances(data, data.numInstances());
    for (int i = 0; i < data.numInstances(); i++) {
      if (whichSubset(data.instance(i)) > -1) {
        insts.add(data.instance(i));
      }
    }
    myDistribution newD = new myDistribution(insts, this);
    newD.addInstWithUnknown(data, m_attIndex);
    dist = newD;
  }

  public final double [] Weights(Instance instance) {
    
    double [] weights;
    int i;
    
    if (instance.isMissing(m_attIndex)) {
      weights = new double [nbSubset];
      for (i=0;i<nbSubset;i++)
      {
        weights [i] = dist.perBag(i)/dist.total();    
      }
      return weights;
    } else {
      return null;
    }
  }

  public final int whichSubset(Instance instance) throws Exception {
    
    if (instance.isMissing(m_attIndex))
    {
      return -1;
    }
    else{
      if (instance.attribute(m_attIndex).isNominal()) {
        return (int)instance.value(m_attIndex);    
      } else {
        if (Utils.smOrEq(instance.value(m_attIndex),m_splitPoint))
        {
          return 0;    
        } else {
          return 1;        
        }
      }
    }
  }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 *
 * @author asus
 */
public class PTRCoba {
    
    public static void main(String[] args){
        Neuron neuron = new Neuron();
        List<Double> inputList = new ArrayList<>();
        List<Double> target = new ArrayList<>();
        int juminput;
        double input;
        int currEpoch = 1;
        int maxEpoch;
        double error;
        double learningRate;
        double momentum;
        
        Scanner reader = new Scanner(System.in);
        
        System.out.println("Max epoch: ");
        maxEpoch = reader.nextInt();
        
        System.out.println("Learning rate: ");
        learningRate = reader.nextDouble();
        
        System.out.println("Momentum: ");
        momentum = reader.nextDouble();
        
        System.out.println("Jumlah input: ");
        juminput = reader.nextInt();
        
        int jumtarget;
        List<Double> targetList = new ArrayList<>();
        
        System.out.println("Jumlah target: ");
        jumtarget = reader.nextInt();
        
        for(int j=0;j<juminput*jumtarget;j++){
            inputList.add((double)0);
        }
        
        for(int i=0;i<jumtarget;i++){
            System.out.println("Target "+(i+1)+" : ");
            input = reader.nextDouble();
            targetList.add(input);
        }
        neuron.initPreviousWeight(juminput,0);
        neuron.initCurrentWeight(juminput,0);
        neuron.initDeltaWeight(juminput,0);
        //neuron.randomizeCurrentWeight();
        neuron.setCurrentWeightZero();
        
        double sum;
        
        List<Double> errorSquareIterationAfterEpoch = new ArrayList<>();
        for(int i=0;i<jumtarget;i++){
            errorSquareIterationAfterEpoch.add((double)0);
        }
        int errorEpoch = 0;
        
        int iteration;
        int h=0;
        do{
            //1 Epoch
            for(int i=0;i<targetList.size();i++){
                sum = 0;
                System.out.println("epoch="+(h+1));
                System.out.println("iteration="+(i+1));
                //Hitung sum
                if(h==0){
                    for(int j=i*juminput;j<(i+1)*juminput;j++){
                        System.out.println("Input " + j + " : ");
                        input = reader.nextDouble();
                        inputList.set(j, input);
                    }
                }
                for(int j=i*juminput;j<(i+1)*juminput;j++){
                    System.out.println("neuron="+neuron.getCurrentWeight().get(j%4));
                    sum = sum + (inputList.get(j)*neuron.getCurrentWeight().get(j%4));
                }
                System.out.println("sum="+sum);
                error = targetList.get(i) - signFunction(sum);
                System.out.println("targetlist="+targetList.get(i));
                System.out.println("sign="+signFunction(sum));
                System.out.println("error="+error);
                for(int j=0;j<juminput;j++){
                    System.out.println("SetNewDeltaW"+j+"= "+inputList.get(j)+"*"+error+"*"+"learningRate");
                    neuron.setDeltaWeight(j, inputList.get(j+(i*juminput))*error*learningRate);
                    neuron.setCurrentWeight(j, neuron.getCurrentWeight().get(j)+neuron.getDeltaWeight().get(j));
                    //System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                    System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                }
            }
            double tempSum;
            double tempSumError = 0;
            for(int i=0;i<jumtarget;i++){
                tempSum = 0;
                for(int j=0;j<juminput;j++){
                    tempSum = tempSum + inputList.get(j+(i*juminput));
                }
//                errorSquareIterationAfterEpoch.set(i, Math.pow(targetList.get(i)-signFunction(tempSum),2));
                tempSumError = tempSumError + Math.pow(targetList.get(i)-signFunction(tempSum),2);
                errorEpoch = (int) (tempSumError * 0.5);
            }
            h++;
        }while(h<=maxEpoch && errorEpoch!=0);
    }
    
    public static int signFunction(double sum){
        int signResult = 0;
        if(sum>=0){
            signResult = 1;
        } else {
            signResult = -1;
        }
        return signResult;
    } 
}

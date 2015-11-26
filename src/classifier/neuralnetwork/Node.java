/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 *
 * @author LINDA
 */
public class Node {
    private float nodeVal;                      //nilai pada node
    private List nodeIn = new ArrayList();      //list node yang menjadi input
    private List nodeOut = new ArrayList();     //list node yang menjadi output
    private  float weight;                      //nilai weight node terhadap node output
    
    public Node(){
        
    }
    
    /*Getter*/
    public float getNodeVal(){                  //mengembalikan nilai nodeVal
        return nodeVal;         
    }
    public Node getNodeIn(int i){               //mengembalikan object sebelum node
        Object object = nodeIn.get(i);
        Node node = (Node) object;
        return node;
    }
    public float getWeight(int i){              //mengembalikan weight object sebelum node ke node
        Object object = nodeIn.get(i);
        Node node = (Node) object;
        return node.weight;
    }
    
    /*Setter*/
    public void setNodeVal(float nodeVal){      //set nilai node
        this.nodeVal = nodeVal;        
    }
    public void setNodeIn(Node node){           //set node sebelum node 
        this.nodeIn.add(node);
    }
    public void setWeight(float weight){        //set weight pada node
        this.weight = weight;
    }
    
    /*Compute*/
    public float countOut(){    //menghitung output
        float output = 0;
        Iterator iterator = nodeIn.iterator();
        while(iterator.hasNext()){
            Object object = iterator.next();
            Node next = (Node) object;
            output += next.nodeVal*next.weight;
        }
        return output;
    }
}

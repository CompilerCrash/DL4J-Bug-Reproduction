import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDBase;

public class ScatterUpdate {

    public static void main(String[] args) {
        INDArray v1 = Nd4j.zeros(5);
        INDArray v2 = new NDBase().scatterUpdate(v1, Nd4j.scalar(2), Nd4j.scalar(1f)); // throws exception
        System.out.println(v2); // expected result: [0, 0, 1, 0, 0]
    }
}

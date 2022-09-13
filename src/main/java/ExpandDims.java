import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ExpandDims {

    public static void main(String[] args) {
        expand2dTo3d();
        System.out.println("=".repeat(40));
        expandLastAxis();
    }

    public static void expand2dTo3d() {
        SameDiff sd = SameDiff.create();
        SDVariable v1 = sd.zero(null, 1, 1);
        SDVariable v2 = sd.expandDims(v1, -1);
        System.out.println(v1.shape().eval()); // [1, 1]
        System.out.println(v2.shape().eval()); // should be [1, 1, 1] but is [1, 1]
    }

    public static void expandLastAxis() {
        INDArray v1 = Nd4j.scalar(0);
        System.out.println(java.util.Arrays.toString(v1.shape())); // []
        INDArray v2 = Nd4j.expandDims(v1, -1); // throws exception
        System.out.println(java.util.Arrays.toString(v2.shape())); // should now be [1]
    }
}

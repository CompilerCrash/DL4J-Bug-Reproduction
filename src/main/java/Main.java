import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {

    public static void main(String[] args) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant(Nd4j.createFromArray(true, false, false, true).reshape(2, 2));
        SDVariable b = sd.constant(Nd4j.createFromArray(false, false).reshape(1, 2));
        SDVariable result = sd.math().or(a, b);
        INDArray eval = result.eval();
        System.out.println(eval);
        /*
        expected result:
        [[      true,     false],
         [     false,      true]]

        actual result:
        [[      true,     false],
         [      true,      true]]
        */
    }
}

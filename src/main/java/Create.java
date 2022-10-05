import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;

public class Create {

    public static void main(String[] args) {
        SameDiff sd = SameDiff.create();
        SDVariable var = sd.create(null, sd.constant(8), DataType.INT32);
        System.out.println(var.dataType()); // should be INT32 but is FLOAT
        System.out.println(var.eval()); // should be uninitialized but is (seemingly) zero-initialized

        System.out.println(var.shape().eval()); // [8]
        var = var.castTo(DataType.BOOL);
        System.out.println(var.shape().eval()); // throws exception (shape op requires numerical type)
    }
}

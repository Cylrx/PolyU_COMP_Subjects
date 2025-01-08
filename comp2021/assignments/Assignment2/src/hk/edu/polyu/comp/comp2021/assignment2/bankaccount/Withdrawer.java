package hk.edu.polyu.comp.comp2021.assignment2.bankaccount;

// A withdrawer that withdraws 'amount' for 'nbrRepetitions' times.
public class Withdrawer implements Runnable{
    private BankAccount ba;
    private int amount;
    private int nbrRepetitions;

    public Withdrawer(BankAccount ba, int amount, int nbrRepetitions){
        if(ba == null || amount <= 0 || nbrRepetitions <= 0)
            throw new IllegalArgumentException();

        this.ba = ba;
        this.amount = amount;
        this.nbrRepetitions = nbrRepetitions;
    }

    public void run() {
        for (int i = 0; i < nbrRepetitions; i++){
			// Note that a withdraw is only allowed when the 
			// balance is greater than the amount to withdraw
            ba.withdraw(amount);       
        }
    }
}

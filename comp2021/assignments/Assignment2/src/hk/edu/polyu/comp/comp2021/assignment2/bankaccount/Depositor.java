package hk.edu.polyu.comp.comp2021.assignment2.bankaccount;

// A depositor that deposits 'amount' for 'nbrRepetitions' times.
public class Depositor implements Runnable{
    private BankAccount ba;
    private int amount;
    private int nbrRepetitions;

    public Depositor(BankAccount ba, int amount, int nbrRepetitions){
        if(ba == null || amount <= 0 || nbrRepetitions <= 0)
            throw new IllegalArgumentException();

        this.ba = ba;
        this.amount = amount;
        this.nbrRepetitions = nbrRepetitions;
    }

    public void run() {
        for(int i = 0; i < nbrRepetitions; i++){
			// A deposit is always allowed.
            ba.deposit(amount); 
        }
    }
}


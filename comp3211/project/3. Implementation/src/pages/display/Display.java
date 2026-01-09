package pages.display;

import model.Board;
import model.Piece;
import model.Square;
import model.elements.Coordinate;
import model.elements.PieceType;
import model.elements.SquareType;

public class Display {
    public static void displayBoard(Board board) {
        if (board == null) {
            System.out.println("Board is null!");
            return;
        }

        int length = board.getLength();
        int width = board.getWidth();

        // Unicode box-drawing characters for better visuals
        final String VERTICAL = "│";  // U+2502
        final String HORIZONTAL = "─";  // U+2500
        final String LEFT_T = "├";     // U+251C (left T-junction)
        final String RIGHT_T = "┤";    // U+2524 (right T-junction)
        final String CROSS = "┼";      // U+253C (cross/junction)
        
        // Print row numbers header (x-axis: A-G) - shows which row each column represents
        // These align with the 7 columns of cells
        // Match the board row prefix format: %3d │ = 5 characters (e.g., "  9 │")
        // Use 5-character wide cells for more square-like appearance
        System.out.print("    " + VERTICAL);
        for (int x = 1; x <= length; x++) {
            String letter = rowNumberToLetter(x);
            System.out.printf("%5s" + VERTICAL, "  " + letter + "  ");  // Format as 5 chars (2 spaces + letter + 2 spaces) then vertical line
        }
        System.out.println();

        // Print top border - aligns with the 7 cell columns (5 chars wide)
        System.out.print("    " + VERTICAL);
        for (int x = 1; x <= length; x++) {
            System.out.print(HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + VERTICAL);
        }
        System.out.println();

        // Print board rows (from top to bottom, which is y=9 to y=1)
        // Note: Board coordinates are (x, y) where x is row (1-7) and y is column (1-9)
        // Left side shows column number (y), right side shows column number (y)
        // Use 5-character wide cells for more square-like appearance
        for (int y = width; y >= 1; y--) {
            // Print cell content row
            System.out.printf("%3d " + VERTICAL, y);
            for (int x = 1; x <= length; x++) {
                Square square = board.getSquare(new Coordinate(x, y));
                Piece piece = square.getPiece();
                SquareType squareType = square.getType();

                String cellContent = formatCell(squareType, piece);
                // Print cell content with vertical separator - content is 3 chars, pad to 5 chars to match header
                System.out.printf("%5s" + VERTICAL, cellContent);
            }
            System.out.printf(" %3d%n", y);
            
            // Add horizontal separator between rows (except after the last row)
            if (y > 1) {
                System.out.print("    " + LEFT_T);
                for (int x = 1; x <= length; x++) {
                    if (x < length) {
                        System.out.print(HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + CROSS);
                    } else {
                        // Last column uses RIGHT_T instead of CROSS
                        System.out.print(HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + RIGHT_T);
                    }
                }
                System.out.println();
            }
        }

        // Print bottom border - aligns with the 7 cell columns (5 chars wide)
        System.out.print("    " + VERTICAL);
        for (int x = 1; x <= length; x++) {
            System.out.print(HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + HORIZONTAL + VERTICAL);
        }
        System.out.println();

        // Print row numbers footer (x-axis: A-G) - these align with the 7 columns of cells
        System.out.print("    " + VERTICAL);
        for (int x = 1; x <= length; x++) {
            String letter = rowNumberToLetter(x);
            System.out.printf("%5s" + VERTICAL, "  " + letter + "  ");  // Format as 5 chars (2 spaces + letter + 2 spaces) then vertical line
        }
        System.out.println();
    }

    private static String formatCell(SquareType squareType, Piece piece) {
        // Unicode symbols for special squares (using characters well-supported in Windows console)
        final String RIVER_SYMBOL = "≈";  // U+2248 (almost equal to, looks like waves)
        final String TRAP_SYMBOL = "■";   // U+25A0 (black square, more compatible than warning)
        final String DEN_SYMBOL = "▲";    // U+25B2 (black triangle, represents den/fortress)
        
        // Format cell based on square type and piece presence
        // Cells are now 5 characters wide for more square-like appearance
        if (piece != null && piece.getStatus()) {
            String pieceSymbol = getPieceSymbol(piece);
            // Combine square type and piece
            if (squareType == SquareType.River) {
                return " ≈" + pieceSymbol + "≈ ";  // 5 chars: space + bracket + symbol + bracket + space
            } else if (squareType == SquareType.Trap) {
                return " " + TRAP_SYMBOL + pieceSymbol + TRAP_SYMBOL + " ";  // 5 chars
            } else if (squareType == SquareType.Den) {
                return " " + DEN_SYMBOL + pieceSymbol + DEN_SYMBOL + " ";  // 5 chars
            } else {
                return "  " + pieceSymbol + "  ";  // 5 chars: 2 spaces + symbol + 2 spaces
            }
        } else {
            // No piece, just show square type
            if (squareType == SquareType.River) {
                return " ≈" + RIVER_SYMBOL + "≈ ";  // 5 chars
            } else if (squareType == SquareType.Trap) {
                return " " + TRAP_SYMBOL + " " + TRAP_SYMBOL + " ";  // 5 chars
            } else if (squareType == SquareType.Den) {
                return " " + DEN_SYMBOL + " " + DEN_SYMBOL + " ";  // 5 chars
            } else {
                return "     ";  // 5 spaces
            }
        }
    }

    private static String getPieceSymbol(Piece piece) {
        PieceType type = piece.getType();
        boolean belongs = piece.getBelongs(); // false = lower player, true = upper player

        // Use uppercase for upper player, lowercase for lower player
        String baseSymbol = getPieceAbbreviation(type);
        return belongs ? baseSymbol.toUpperCase() : baseSymbol.toLowerCase();
    }

    private static String rowNumberToLetter(int rowNumber) {
        // Convert row number (1-7) to letter (A-G)
        return String.valueOf((char) ('A' + rowNumber - 1));
    }

    private static String getPieceAbbreviation(PieceType type) {
        switch (type) {
            case Rat:
                return "r";
            case Cat:
                return "c";
            case Dog:
                return "d";
            case Wolf:
                return "w";
            case Leopard:
                return "p";  // "p" for leopard (becomes "P" for upper player)
            case Tiger:
                return "t";
            case Lion:
                return "i";  // "i" for Lion (becomes "I" for upper player)
            case Elephant:
                return "e";
            default:
                return "?";
        }
    }

    public static void displayBoardWithLegend(Board board) {
        displayBoard(board);
        System.out.println("\nLegend:");
        System.out.println("Pieces: Lower player (lowercase), Upper player (UPPERCASE)");
        System.out.println("  r/R = Rat, c/C = Cat, d/D = Dog, w/W = Wolf");
        System.out.println("  p/P = Leopard, t/T = Tiger, i/I = Lion, e/E = Elephant");
        System.out.println("Squares: ≈≈≈ = River, ■ = Trap, ▲ = Den, (space) = Normal");
        System.out.println("Coordinates: (column, row) where column A-G (left to right), row 1-9 (bottom to top)");
    }
}
    